"""
NewAPI OpenAI 兼容图像生成插件
支持自定义 Base URL 和 API Key，自动从 /v1/models 获取可用模型列表。

插件目录结构：
  newapi_openai/
  ├── main.py          ← 生成逻辑 + 模型获取
  ├── ui/
  │   └── index.html   ← 前端设置界面（iframe 加载）
  └── info.json        ← 插件元信息
"""

import base64
import json
import os
import shutil
import sys
import tempfile
import traceback
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import urlparse
import requests
from PIL import Image, ImageDraw, ImageOps

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

plugin_dir = Path(__file__).parent
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from plugin_utils import load_plugin_config, update_plugin_params

_PLUGIN_FILE = __file__
_PLUGIN_VERSION = "1.2.11"
_UPDATE_REPO = "liushuai0109001-cell/huiju-openai-image-plugin"

_FACE_CASCADE = None
_EYE_CASCADE = None
_EYE_CASCADE_FALLBACK = None
_PROFILE_FACE_CASCADE = None
_CHARACTER_IMAGE_ID_PATTERN = re.compile(r"^(?:\d+_)?character_\d+(?:_|$)", re.IGNORECASE)
_CHARACTER_ROLE_KEYWORDS = (
    "角色设定", "人物设定", "角色图", "角色卡", "角色三视图", "角色四视图",
    "人物三视图", "人物四视图", "三视图", "四视图", "角色参考图", "角色形象",
    "人物形象", "角色立绘", "人物立绘", "角色设计", "角色档案", "人设图",
    "character design", "character sheet", "character turnaround", "turnaround sheet",
    "character reference", "model sheet", "character portrait",
)

# ===================== 默认参数 =====================

_DEFAULT_PARAMS = {
    "api_key": "",
    "base_url": "https://api.openai.com",
    "model": "dall-e-3",
    "size": "1024x1024",
    "aspect_ratio": "1:1",
    "image_size": "1K",
    "n": 1,
    "quality": "standard",
    "style": "vivid",
    "response_format": "url",
    "timeout": 300,
    "enable_face_processing": True,
    "update_repo": _UPDATE_REPO,
    "update_asset_name": "",
}

# ===================== 尺寸预设（按模型） =====================

_SIZE_OPTIONS = {
    "dall-e-2": ["256x256", "512x512", "1024x1024"],
    "dall-e-3": ["1024x1024", "1792x1024", "1024x1792"],
    "gpt-image": ["1024x1024", "1536x1024", "1024x1536"],
    "gpt-image-2": ["1K", "2K", "4K"],
    "gemini": ["1K", "2K", "4K"],
}

_SIZE_LABELS = {
    "256x256": "256x256（小方形）",
    "512x512": "512x512（中方块）",
    "1024x1024": "1024x1024（方形）",
    "1792x1024": "1792x1024（宽横屏）",
    "1024x1792": "1024x1792（宽竖屏）",
    "1536x1024": "1536x1024（横屏）",
    "1024x1536": "1024x1536（竖屏）",
}


# ===================== 工具函数 =====================

def _get_size_options_for_model(model: str) -> list:
    """根据模型名称返回可用的尺寸选项。"""
    for key, sizes in _SIZE_OPTIONS.items():
        if key in model.lower() or model.lower() in key:
            return sizes
    return ["1024x1024", "1792x1024", "1024x1792"]


def _is_firefly_model(model: str) -> bool:
    return "firefly" in str(model or "").lower()


def _supports_images_generation_refs(model: str) -> bool:
    """Models that accept reference images on /v1/images/generations."""
    model_l = str(model or "").lower()
    return any(key in model_l for key in ("firefly", "gpt-image", "adobe"))


def _gpt_image_quality_for_request(model: str, image_size: str, quality: str) -> str:
    """Return per-request GPT Image quality only for 1K/2K Adobe GPT Image models."""
    model_l = str(model or "").lower()
    if "firefly" not in model_l and "gpt-image" not in model_l:
        return ""
    if str(image_size or "").upper() not in {"1K", "2K"}:
        return ""
    quality_l = str(quality or "").strip().lower()
    if quality_l in {"low", "medium", "high"}:
        return quality_l
    if quality_l == "hd":
        return "high"
    if quality_l == "standard":
        return "medium"
    return "medium"


_GPT_IMAGE_SIZE_MAP: Dict[Tuple[str, str], str] = {
    ("1:1", "1K"): "1024x1024",
    ("1:1", "2K"): "2048x2048",
    ("1:1", "4K"): "4096x4096",
    ("16:9", "1K"): "1536x864",
    ("16:9", "2K"): "2048x1152",
    ("16:9", "4K"): "4096x2304",
    ("9:16", "1K"): "864x1536",
    ("9:16", "2K"): "1152x2048",
    ("9:16", "4K"): "2304x4096",
    ("4:3", "1K"): "1408x1056",
    ("4:3", "2K"): "2816x2112",
    ("4:3", "4K"): "3840x2880",
    ("3:4", "1K"): "1056x1408",
    ("3:4", "2K"): "2112x2816",
    ("3:4", "4K"): "2880x3840",
}


def _normalize_aspect_ratio_value(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    compact = re.sub(r"\s+", "", raw).replace("*", "x").replace("×", "x")
    if "16:9" in compact or "16x9" in compact or "landscape" in compact:
        return "16:9"
    if "9:16" in compact or "9x16" in compact or "portrait" in compact or "vertical" in compact:
        return "9:16"
    if "4:3" in compact or "4x3" in compact:
        return "4:3"
    if "3:4" in compact or "3x4" in compact:
        return "3:4"
    if "1:1" in compact or "1x1" in compact or "square" in compact:
        return "1:1"
    match = re.search(r"(\d{3,5})x(\d{3,5})", compact)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
        if width == height:
            return "1:1"
        ratio = width / max(1, height)
        if ratio > 1.55:
            return "16:9"
        if ratio < 0.72:
            return "9:16"
        return "4:3" if width > height else "3:4"
    return raw if raw in {"1:1", "16:9", "9:16", "4:3", "3:4"} else ""


def _normalize_image_size_value(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    compact = re.sub(r"\s+", "", raw)
    if compact in {"4k", "uhd", "ultra", "high"}:
        return "4K"
    if compact in {"2k", "hd", "medium"}:
        return "2K"
    if compact in {"1k", "sd", "low", "standard"}:
        return "1K"
    match = re.search(r"(\d{3,5})x(\d{3,5})", compact)
    if match:
        long_edge = max(int(match.group(1)), int(match.group(2)))
        if long_edge >= 3500:
            return "4K"
        if long_edge >= 1500:
            return "2K"
        return "1K"
    if compact in {"4", "4000", "4096"}:
        return "4K"
    if compact in {"2", "2000", "2048"}:
        return "2K"
    if compact in {"1", "1000", "1024"}:
        return "1K"
    return ""


def _gpt_image_size(aspect_ratio: str, image_size: str) -> str:
    return (_GPT_IMAGE_SIZE_MAP.get((aspect_ratio, image_size))
            or _GPT_IMAGE_SIZE_MAP.get((aspect_ratio, "1K"))
            or "1024x1024")


def _resolution_fields(aspect_ratio: str, image_size: str) -> Dict[str, Any]:
    resolution = _normalize_image_size_value(image_size) or "1K"
    resolution_lower = resolution.lower()
    size_value = _gpt_image_size(aspect_ratio, resolution)
    image_config = {"aspectRatio": aspect_ratio, "imageSize": resolution, "outputResolution": resolution}
    extra_body = {
        "aspect_ratio": aspect_ratio, "aspectRatio": aspect_ratio, "ratio": aspect_ratio,
        "quality": resolution_lower, "resolution": resolution,
        "output_resolution": resolution, "outputResolution": resolution,
        "image_size": resolution, "imageSize": resolution, "size": size_value,
        "imageConfig": image_config, "generationConfig": {"imageConfig": image_config},
    }
    return {
        "size": size_value, "aspect_ratio": aspect_ratio, "aspectRatio": aspect_ratio,
        "ratio": aspect_ratio, "quality": resolution_lower, "resolution": resolution,
        "output_resolution": resolution, "outputResolution": resolution,
        "image_size": resolution, "imageSize": resolution,
        "generationConfig": {"imageConfig": image_config}, "extra_body": extra_body,
    }


def _apply_weigrok_resolution_payload(payload: Dict[str, Any], aspect_ratio: str, image_size: str) -> None:
    payload.update(_resolution_fields(aspect_ratio, image_size))


def _get_model_from_config() -> str:
    """从配置中读取当前模型。"""
    params = _DEFAULT_PARAMS.copy()
    params.update(load_plugin_config(_PLUGIN_FILE))
    return params.get("model", "dall-e-3")


def _debug_log(msg: str) -> None:
    try:
        log_dir = plugin_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"debug_{datetime.now().strftime('%Y%m%d')}.log"
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass


# ===================== 角色图自动人脸处理 =====================

def _version_tuple(value: str):
    text = str(value or "").strip().lstrip("vV")
    parts = []
    for item in text.replace("-", ".").split("."):
        try:
            parts.append(int("".join(ch for ch in item if ch.isdigit()) or "0"))
        except Exception:
            parts.append(0)
    return tuple((parts + [0, 0, 0])[:3])


def _normalize_update_repo(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("https://github.com/"):
        text = text[len("https://github.com/"):]
    return text.strip("/ ")


def _get_latest_release(repo: str, timeout: int = 20) -> dict:
    repo = _normalize_update_repo(repo)
    if not repo or "/" not in repo:
        return {"ok": False, "error": "插件内置更新源无效"}
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "huiju-image-plugin-updater"}
    release_url = f"https://api.github.com/repos/{repo}/releases/latest"
    release_resp = requests.get(release_url, headers=headers, timeout=timeout, proxies={"http": None, "https": None})
    release = release_resp.json() if release_resp.status_code == 200 else {}

    tags_url = f"https://api.github.com/repos/{repo}/tags?per_page=30"
    tags_resp = requests.get(tags_url, headers=headers, timeout=timeout, proxies={"http": None, "https": None})
    tags = tags_resp.json() if tags_resp.status_code == 200 else []
    valid_tags = [item for item in tags if isinstance(item, dict) and item.get("name") and item.get("zipball_url")]
    latest_tag = max(valid_tags, key=lambda item: _version_tuple(item.get("name")), default={})

    release_tag = str(release.get("tag_name") or "")
    tag_name = str(latest_tag.get("name") or "")
    if tag_name and (not release_tag or _version_tuple(tag_name) > _version_tuple(release_tag)):
        latest = tag_name.lstrip("vV")
        return {
            "ok": True,
            "repo": repo,
            "current_version": _PLUGIN_VERSION,
            "latest_version": latest,
            "has_update": _version_tuple(latest) > _version_tuple(_PLUGIN_VERSION),
            "release_name": tag_name,
            "html_url": f"https://github.com/{repo}/tree/{tag_name}",
            "assets": [{"name": f"{tag_name}.zip", "download_url": latest_tag["zipball_url"]}],
        }
    if not release:
        error_status = release_resp.status_code if release_resp.status_code != 404 else tags_resp.status_code
        return {"ok": False, "error": f"GitHub 更新源不可用，HTTP {error_status}"}

    latest = str(release.get("tag_name") or "").lstrip("v")
    assets = [
        {"name": item.get("name") or "", "download_url": item.get("browser_download_url") or ""}
        for item in (release.get("assets") or [])
        if item.get("browser_download_url")
    ]
    if not assets and release.get("zipball_url"):
        assets.append({"name": f"{release.get('tag_name') or 'source'}.zip", "download_url": release.get("zipball_url")})
    return {
        "ok": True,
        "repo": repo,
        "current_version": _PLUGIN_VERSION,
        "latest_version": latest,
        "has_update": bool(latest and _version_tuple(latest) > _version_tuple(_PLUGIN_VERSION)),
        "release_name": release.get("name") or release.get("tag_name") or "",
        "html_url": release.get("html_url") or "",
        "assets": assets,
    }


def _choose_update_asset(release: dict, preferred_name: str = "") -> dict:
    assets = release.get("assets") or []
    preferred = str(preferred_name or "").strip()
    if preferred:
        for asset in assets:
            if asset.get("name") == preferred:
                return asset
    zip_assets = [asset for asset in assets if str(asset.get("name") or "").lower().endswith(".zip")]
    if zip_assets:
        return zip_assets[0]
    return assets[0] if assets else {}


def _copy_plugin_update(source_dir: Path) -> Path:
    backup_dir = plugin_dir.parent / f"{plugin_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ignore = shutil.ignore_patterns(".git", "__pycache__", "logs", "*.pyc", "*.pyo")
    shutil.copytree(plugin_dir, backup_dir, ignore=ignore)
    for item in source_dir.iterdir():
        if item.name in {".git", "__pycache__", "logs"}:
            continue
        target = plugin_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True, ignore=ignore)
        else:
            shutil.copy2(item, target)
    return backup_dir


def _apply_github_update(repo: str, preferred_asset_name: str = "") -> dict:
    release = _get_latest_release(repo)
    if not release.get("ok"):
        return release
    if not release.get("has_update"):
        return {"ok": True, "updated": False, **release}
    asset = _choose_update_asset(release, preferred_asset_name)
    if not asset:
        return {"ok": False, "error": "最新 Release 没有可下载的 zip 资源", **release}

    with tempfile.TemporaryDirectory(prefix="huiju_image_plugin_update_") as temp_name:
        temp_dir = Path(temp_name)
        zip_path = temp_dir / (asset.get("name") or "update.zip")
        _debug_log(f"[update] downloading {asset.get('download_url')}")
        with requests.get(asset["download_url"], stream=True, timeout=180, proxies={"http": None, "https": None}) as resp:
            if resp.status_code != 200:
                return {"ok": False, "error": f"下载失败 HTTP {resp.status_code}: {resp.text[:200]}", **release}
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 512):
                    if chunk:
                        fh.write(chunk)

        extract_dir = temp_dir / "extract"
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)

        candidates = [extract_dir]
        candidates.extend([p for p in extract_dir.rglob("*") if p.is_dir()])
        source_dir = None
        for candidate in candidates:
            if (candidate / "main.py").exists() and (candidate / "ui" / "index.html").exists():
                source_dir = candidate
                break
        if source_dir is None:
            return {"ok": False, "error": "更新包内没有找到插件 main.py 和 ui/index.html", **release}

        backup_dir = _copy_plugin_update(source_dir)
        update_plugin_params(_PLUGIN_FILE, {
            "update_repo": _normalize_update_repo(repo),
            "update_asset_name": asset.get("name") or "",
            "last_update_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        return {"ok": True, "updated": True, "backup_dir": str(backup_dir), "asset": asset.get("name") or "", **release}


def _is_character_role_request(context: Dict[str, Any], prompt: str) -> bool:
    """Identify character images by 字字's character_N identifier, then by prompt."""
    for key in ("face_processing", "enable_face_processing", "is_character_role"):
        if context.get(key) is True:
            return True
        if context.get(key) is False:
            return False

    # 字字角色图使用 character_13、character_14 ... 这样的稳定编号；
    # 普通分镜图不会使用这个编号，故它比自然语言提示词可靠得多。
    unique_name = str(context.get("unique_name") or "").strip()
    if _CHARACTER_IMAGE_ID_PATTERN.match(unique_name):
        return True

    text_parts = [str(prompt or "")]
    for key in ("task_type", "image_type", "generation_type", "title", "name", "unique_name"):
        value = context.get(key)
        if isinstance(value, str):
            text_parts.append(value)
    request_text = "\n".join(text_parts).lower()
    return any(keyword.lower() in request_text for keyword in _CHARACTER_ROLE_KEYWORDS)


def _setting_enabled(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    if normalized in {"false", "0", "no", "off", "disabled", "关闭"}:
        return False
    if normalized in {"true", "1", "yes", "on", "enabled", "开启"}:
        return True
    return default


def _clamp_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(value) for value in box)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    box_w, box_h = max(1, x1 - x0), max(1, y1 - y0)
    x0 = max(0, min(x0, max(0, width - 1)))
    y0 = max(0, min(y0, max(0, height - 1)))
    x1 = min(width, x0 + box_w)
    y1 = min(height, y0 + box_h)
    x0 = max(0, x1 - box_w)
    y0 = max(0, y1 - box_h)
    return x0, y0, x1, y1


def _get_face_cascades():
    """Load bundled OpenCV Haar models once; return None when unavailable."""
    global _FACE_CASCADE, _EYE_CASCADE, _EYE_CASCADE_FALLBACK, _PROFILE_FACE_CASCADE
    if cv2 is None or np is None:
        return None
    if _FACE_CASCADE is False:
        return None
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE, _EYE_CASCADE, _EYE_CASCADE_FALLBACK, _PROFILE_FACE_CASCADE
    try:
        cascade_dir = cv2.data.haarcascades
        # OpenCV 4.8 on Windows cannot open its XML models through this app's
        # Chinese installation path. Python can copy the same bundled models to
        # an ASCII cache path, which lets CascadeClassifier load them normally.
        cache_dir = Path(tempfile.gettempdir()) / "zz_face_plugin_cascades"
        if any(ord(char) > 127 for char in str(cache_dir)):
            cache_dir = Path(os.environ.get("SystemDrive", "C:")) / "zz_face_plugin_cascades"
        cache_dir.mkdir(parents=True, exist_ok=True)

        def _cached_model(name: str) -> str:
            source = Path(cascade_dir) / name
            destination = cache_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"缺少 OpenCV 模型文件: {source}")
            if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
                shutil.copyfile(source, destination)
            return str(destination)

        face = cv2.CascadeClassifier(_cached_model("haarcascade_frontalface_alt2.xml"))
        eye = cv2.CascadeClassifier(_cached_model("haarcascade_eye_tree_eyeglasses.xml"))
        eye_fallback = cv2.CascadeClassifier(_cached_model("haarcascade_eye.xml"))
        profile = cv2.CascadeClassifier(_cached_model("haarcascade_profileface.xml"))
        if face.empty() or eye.empty():
            raise RuntimeError("OpenCV Haar cascade files are unavailable")
        _FACE_CASCADE, _EYE_CASCADE = face, eye
        _EYE_CASCADE_FALLBACK, _PROFILE_FACE_CASCADE = eye_fallback, profile
        return face, eye, eye_fallback, profile
    except Exception as exc:
        _FACE_CASCADE = False
        _debug_log(f"角色图人脸处理不可用: {exc}")
        return None


def _nms_face_boxes(boxes: list, threshold: float = 0.35) -> list:
    """Remove duplicate front/profile detections while preserving separate poses."""
    kept = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        x, y, box_w, box_h = box
        duplicate = False
        for other_x, other_y, other_w, other_h in kept:
            intersect_x0, intersect_y0 = max(x, other_x), max(y, other_y)
            intersect_x1 = min(x + box_w, other_x + other_w)
            intersect_y1 = min(y + box_h, other_y + other_h)
            intersection = max(0, intersect_x1 - intersect_x0) * max(0, intersect_y1 - intersect_y0)
            union = box_w * box_h + other_w * other_h - intersection
            if union and intersection / union >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def _detect_role_faces(image: Image.Image):
    cascades = _get_face_cascades()
    if cascades is None:
        return None, []
    face_cascade, _, _, profile_cascade = cascades
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    height, width = gray.shape[:2]
    minimum = max(24, min(width, height) // 35)
    found = face_cascade.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=4, minSize=(minimum, minimum), flags=cv2.CASCADE_SCALE_IMAGE
    )
    if found is None or len(found) == 0:
        found = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(minimum, minimum))
    faces = [tuple(map(int, item)) for item in found] if found is not None else []
    if profile_cascade is not None and not profile_cascade.empty():
        for source_gray, flipped in ((gray, False), (cv2.flip(gray, 1), True)):
            profiles = profile_cascade.detectMultiScale(
                source_gray, scaleFactor=1.05, minNeighbors=3, minSize=(minimum, minimum)
            )
            if profiles is None:
                continue
            for x, y, face_w, face_h in profiles:
                x = width - int(x) - int(face_w) if flipped else int(x)
                faces.append((x, int(y), int(face_w), int(face_h)))
    faces = _nms_face_boxes(faces)
    if not faces:
        return gray, []
    # Character sheets commonly place the primary portrait on the left. Prefer it,
    # but never select a tiny detection over a clear main face.
    portrait_end = max(width // 2, int(width * 0.35))
    faces.sort(key=lambda box: box[2] * box[3] * (1.5 if box[0] + box[2] // 2 < portrait_end else 0.5), reverse=True)
    return gray, faces


def _detect_eye_box(gray, face: Tuple[int, int, int, int], image_size: Tuple[int, int]):
    cascades = _get_face_cascades()
    if cascades is None:
        return None
    _, eye_cascade, eye_fallback, _ = cascades
    fx, fy, fw, fh = face
    y0, y1 = fy + int(fh * 0.18), fy + int(fh * 0.58)
    x0, x1 = fx + int(fw * 0.08), fx + int(fw * 0.92)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    found = eye_cascade.detectMultiScale(roi, scaleFactor=1.08, minNeighbors=3, minSize=(12, 12))
    if (found is None or len(found) == 0) and eye_fallback is not None and not eye_fallback.empty():
        found = eye_fallback.detectMultiScale(roi, scaleFactor=1.08, minNeighbors=3, minSize=(12, 12))
    if found is None or len(found) == 0:
        return None
    boxes = [(x0 + int(x), y0 + int(y), x0 + int(x + w), y0 + int(y + h)) for x, y, w, h in found]
    if len(boxes) == 1:
        bx0, by0, bx1, by1 = boxes[0]
        center = fx + fw // 2
        mirrored_center = 2 * center - (bx0 + bx1) // 2
        half = (bx1 - bx0) // 2
        boxes.append((mirrored_center - half, by0, mirrored_center + half, by1))
    return _clamp_box(
        (min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes)),
        *image_size,
    )


def _integral_window_sum(integral, x: int, y: int, width: int, height: int) -> float:
    x1, y1 = x + width, y + height
    return float(integral[y1, x1] - integral[y, x1] - integral[y1, x] + integral[y, x])


def _find_empty_eye_paste_area(
    image: Image.Image,
    paste_size: Tuple[int, int],
    faces: list,
    source_box: Tuple[int, int, int, int],
):
    """Find background with neither a body silhouette nor texture or face overlap."""
    arr = np.asarray(image.convert("RGB"))
    height, width = arr.shape[:2]
    box_w, box_h = paste_size
    if box_w >= width or box_h >= height:
        return None

    border_h, border_w = max(2, height // 30), max(2, width // 30)
    border_pixels = np.concatenate((
        arr[:border_h].reshape(-1, 3), arr[-border_h:].reshape(-1, 3),
        arr[:, :border_w].reshape(-1, 3), arr[:, -border_w:].reshape(-1, 3),
    ), axis=0).astype(np.float32)
    background = np.median(border_pixels, axis=0)
    distance = np.linalg.norm(arr.astype(np.float32) - background, axis=2)
    foreground = (distance > 38).astype(np.uint8)
    foreground = cv2.dilate(foreground, np.ones((5, 5), np.uint8), iterations=1)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = (cv2.Canny(gray, 60, 140) > 0).astype(np.uint8)

    avoid = np.zeros((height, width), dtype=np.uint8)
    for x, y, face_w, face_h in faces:
        center_x = x + face_w / 2
        x0, x1 = max(0, round(center_x - face_w * 1.45)), min(width, round(center_x + face_w * 1.45))
        y0, y1 = max(0, round(y - face_h * 1.2)), min(height, round(y + face_h * 8.0))
        avoid[y0:y1, x0:x1] = 1
    sx0, sy0, sx1, sy1 = source_box
    margin = max(8, box_h // 2)
    avoid[max(0, sy0 - margin):min(height, sy1 + margin), max(0, sx0 - margin):min(width, sx1 + margin)] = 1

    foreground_integral = cv2.integral(foreground)
    edges_integral = cv2.integral(edges)
    avoid_integral = cv2.integral(avoid)
    area = box_w * box_h
    step = max(4, min(box_w, box_h) // 4)
    xs, ys = list(range(0, width - box_w + 1, step)), list(range(0, height - box_h + 1, step))
    if xs[-1] != width - box_w:
        xs.append(width - box_w)
    if ys[-1] != height - box_h:
        ys.append(height - box_h)

    best = None
    best_score = float("inf")
    for y in ys:
        for x in xs:
            if _integral_window_sum(avoid_integral, x, y, box_w, box_h) > 0:
                continue
            foreground_ratio = _integral_window_sum(foreground_integral, x, y, box_w, box_h) / area
            edge_ratio = _integral_window_sum(edges_integral, x, y, box_w, box_h) / area
            score = foreground_ratio * 7.0 + edge_ratio * 2.5 + (1.0 - x / max(1, width - box_w)) * 0.08
            if score < best_score:
                best_score, best = score, (x, y, foreground_ratio, edge_ratio)
    if best is None or best[2] > 0.08 or best[3] > 0.04:
        return None
    return best[0], best[1], best[0] + box_w, best[1] + box_h


def _process_character_role_image(image_path: str) -> bool:
    """Mask every visible pose's eyes and atomically replace the saved file."""
    temporary_path = None
    try:
        if _get_face_cascades() is None:
            return False
        with Image.open(image_path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = source.size
        gray, faces = _detect_role_faces(source)
        if not faces:
            _debug_log(f"角色图人脸处理跳过，未检测到人脸: {image_path}")
            return False

        primary_face = faces[0]

        def _bar_box_for_face(face, detected_box=None):
            face_x, face_y, face_w, face_h = face
            if detected_box is None:
                if face_w < face_h * 0.9:
                    detected_box = (
                        face_x + int(face_w * 0.12), face_y + int(face_h * 0.18),
                        face_x + int(face_w * 0.92), face_y + int(face_h * 0.40),
                    )
                else:
                    detected_box = (
                        face_x + int(face_w * 0.18), face_y + int(face_h * 0.22),
                        face_x + int(face_w * 0.82), face_y + int(face_h * 0.42),
                    )
            left, top, right, bottom = detected_box
            horizontal_padding = max(0, min(20, int(face_w * 0.22)))
            center_y = (top + bottom) // 2 + 8
            adaptive_height = max(12, int(face_h * 0.17))
            bar_height = max(12, min(36, adaptive_height, height))
            return _clamp_box(
                (left - horizontal_padding, center_y - bar_height // 2,
                 right + horizontal_padding, center_y - bar_height // 2 + bar_height),
                width, height,
            )

        primary_eye_box = _detect_eye_box(gray, primary_face, source.size)
        primary_bar_box = _bar_box_for_face(primary_face, primary_eye_box)
        crop = source.crop(primary_bar_box)
        result = source.copy()
        draw = ImageDraw.Draw(result)
        masked_count = 0
        detected_count = 0
        for face in sorted(faces, key=lambda item: (item[0], item[1])):
            detected_eye_box = _detect_eye_box(gray, face, source.size)
            # Side-profile detectors can mistake lower body details for faces.
            if face[1] > height * 0.36 and detected_eye_box is None:
                continue
            bar_box = _bar_box_for_face(face, detected_eye_box)
            bx0, by0, bx1, by1 = bar_box
            if bx1 - bx0 < 4 or by1 - by0 < 4:
                continue
            draw.rectangle((bx0, by0, bx1 - 1, by1 - 1), fill=(0, 0, 0))
            masked_count += 1
            detected_count += int(detected_eye_box is not None)

        paste_box = _find_empty_eye_paste_area(source, crop.size, faces, primary_bar_box)
        if paste_box is not None:
            px0, py0, px1, py1 = paste_box
            result.paste(crop.resize((px1 - px0, py1 - py0), Image.Resampling.LANCZOS), (px0, py0))

        file_dir = os.path.dirname(os.path.abspath(image_path))
        with tempfile.NamedTemporaryFile(prefix=".face-processing-", suffix=".png", dir=file_dir, delete=False) as handle:
            temporary_path = handle.name
        result.save(temporary_path, format="PNG")
        os.replace(temporary_path, image_path)
        temporary_path = None
        place_note = "已粘贴到背景空位" if paste_box is not None else "未找到安全空位，仅遮挡眼睛"
        detection_note = f"处理 {masked_count} 处眼睛（识别 {detected_count} 处，其余使用定位回退）"
        _debug_log(f"角色图人脸处理完成: {image_path}; {detection_note}; {place_note}")
        print(f"  [人脸处理] {detection_note}，{place_note}: {os.path.basename(image_path)}")
        return masked_count > 0
    except Exception as exc:
        _debug_log(f"角色图人脸处理失败，已保留原图: {image_path}; {exc}")
        print(f"  [人脸处理] 失败，保留原图: {exc}")
        return False
    finally:
        if temporary_path:
            try:
                os.remove(temporary_path)
            except OSError:
                pass


# ===================== 模型列表获取 =====================

def _fetch_models_from_api(base_url: str, api_key: str, timeout: int = 15) -> dict:
    """
    从 OpenAI 兼容的 /v1/models 接口获取模型列表。
    
    Returns:
        dict: {"ok": True, "models": [...], "default_model": "..."} 或 {"ok": False, "error": "..."}
    """
    endpoint = f"{base_url.rstrip('/')}/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    try:
        resp = requests.get(endpoint, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:500]
            return {
                "ok": False,
                "error": f"HTTP {resp.status_code}: {detail}",
            }
        
        data = resp.json()
        raw_models = data.get("data", [])
        if not raw_models:
            return {"ok": False, "error": "API 返回了空的模型列表"}
        
        model_ids = sorted([m.get("id", "") for m in raw_models if m.get("id")])
        
        return {
            "ok": True,
            "models": model_ids,
            "default_model": model_ids[0] if model_ids else "dall-e-3",
        }
        
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "请求超时，请检查网络连接"}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "连接失败，请检查 Base URL 是否正确"}
    except Exception as e:
        return {"ok": False, "error": f"获取模型列表失败: {str(e)}"}


# ===================== 参考图处理 =====================

def _read_image_as_base64(image_path: str, max_size: int = 4096) -> str:
    """
    读取图片并转为 data URI 的 base64（用于 chat/completions 多模态入参）。
    适当压缩以避免 payload 过大。
    
    Args:
        image_path: 图片路径
        max_size: 最大边长（默认4096px，4K），超过自动等比缩放
    """
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        from PIL import Image
        import io
        
        ext = os.path.splitext(image_path)[1].lower()
        
        # 打开并压缩图片
        with Image.open(image_path) as img:
            # 转换为RGB
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # 等比缩放到512px
            width, height = img.size
            if width > max_size or height > max_size:
                ratio = min(max_size / width, max_size / height)
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"[荟聚] 图片已压缩: {width}x{height} -> {new_size[0]}x{new_size[1]}")
            
            # JPEG quality=95，画质接近无损
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
        print(f"[荟聚] 参考图已编码: {os.path.basename(image_path)} ({len(b64)//1024}KB base64)")
        return f"data:image/jpeg;base64,{b64}"
    except ImportError:
        # PIL 未安装，回退到原始方式
        print(f"[荟聚] PIL未安装，使用原始图片（可能较大）")
        try:
            ext = os.path.splitext(image_path)[1].lower()
            mime = "image/png" if ext == ".png" else "image/webp" if ext == ".webp" else "image/jpeg"
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{b64}"
        except Exception as e:
            print(f"[荟聚] 读取图片失败 {image_path}: {e}")
            return None
    except Exception as e:
        print(f"[荟聚] 处理图片失败 {image_path}: {e}")
        return None


def _collect_reference_images(reference_images: dict) -> list:
    """收集参考图片路径列表（参考图片MAP + 首帧/尾帧），去重后返回有效路径。"""
    paths = []
    if not reference_images:
        return paths
    
    # 参考图片MAP
    ref_map = reference_images.get("参考图片MAP", {})
    if isinstance(ref_map, dict) and ref_map:
        for key in sorted(ref_map.keys(), key=lambda k: (
            int(k) if isinstance(k, str) and k.isdigit() else 
            int(k) if isinstance(k, int) else float('inf')
        )):
            p = ref_map[key]
            if p:
                clean = str(p).split("?")[0]
                if os.path.exists(clean) and clean not in paths:
                    paths.append(clean)
    
    # 首帧/尾帧
    for key in ["首帧", "尾帧"]:
        p = reference_images.get(key)
        if p:
            clean = str(p).split("?")[0]
            if os.path.exists(clean) and clean not in paths:
                paths.append(clean)
    
    return paths


def _parse_chat_response(result: dict, base_url: str = "") -> list:
    """解析 /v1/chat/completions 响应，提取图片URL（自动补全相对路径）。"""
    images = []
    choices = result.get("choices", [])
    
    # 追加完整响应到日志（截断到500字）
    result_str = json.dumps(result, ensure_ascii=False)
    print(f"  [Chat响应解析] keys: {list(result.keys())}, choices数: {len(choices) if isinstance(choices, list) else 'N/A'}")
    print(f"  [Chat响应] {result_str[:500]}")
    
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message", {})
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        finish_reason = choice.get("finish_reason", "")
        print(f"  [Chat解析] finish_reason={finish_reason}, content类型={type(content).__name__}")
        
        if isinstance(content, str):
            txt = content.strip()
            print(f"  [Chat解析] content前200字: {txt[:200]}")
            if txt.startswith("data:image"):
                images.append({"type": "b64_json", "data": txt.split(",", 1)[1]})
                print(f"  [Chat解析] 提取到 data:image")
            elif txt.startswith("!["):
                # Adobe2api 返回 Markdown 图片格式: ![Generated Image](url)
                import re
                md_urls = re.findall(r'!\[.*?\]\((https?://[^\)]+)\)', txt)
                for u in md_urls:
                    images.append({"type": "url", "data": u})
                    print(f"  [Chat解析] Markdown图片: {u[:100]}")
            elif txt.startswith("/") or txt.startswith("http"):
                images.append({"type": "url", "data": txt})
                print(f"  [Chat解析] 提取到 URL: {txt[:100]}")
            elif txt.startswith("{"):
                try:
                    j = json.loads(txt)
                    if "data" in j:
                        sub = _parse_images_response(j)
                        images.extend(sub)
                        print(f"  [Chat解析] 内嵌JSON data, 提取{len(sub)}张")
                    elif "url" in j:
                        images.append({"type": "url", "data": j["url"]})
                        print(f"  [Chat解析] 内嵌JSON url")
                except Exception:
                    pass
            else:
                # 尝试从文本中正则提取URL（排除末尾括号等非URL字符）
                import re
                found_urls = re.findall(r'https?://[^\s\"\'<>\\)]+', txt)
                for u in found_urls:
                    images.append({"type": "url", "data": u})
                    print(f"  [Chat解析] 正则提取: {u[:100]}")
                    
        elif isinstance(content, list):
            print(f"  [Chat解析] content是列表，{len(content)}个block")
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "image_url":
                    img_url = block.get("image_url", {})
                    if isinstance(img_url, dict):
                        url_val = img_url.get("url", "")
                    else:
                        url_val = str(img_url)
                    if url_val:
                        images.append({"type": "url", "data": url_val})
                        print(f"  [Chat解析] image_url: {url_val[:100]}")
                        
    if not images and "data" in result:
        images = _parse_images_response(result)
        print(f"  [Chat解析] 兜底images格式, 提取{len(images)}张")
        
    # 最后兜底：直接搜索整个result中的generated路径
    if not images:
        import re
        all_urls = re.findall(r'(?:["\']|markdown\)?)((?:https?:)?//[^\s\"\'<>\)]*?/(?:generated|output|images)/[^\s\"\'<>\)]*?\.(?:png|jpg|jpeg|webp))', json.dumps(result, ensure_ascii=False))
        for u in all_urls:
            if not u.startswith("http"):
                u = "https:" + u if u.startswith("//") else u
            images.append({"type": "url", "data": u})
            print(f"  [Chat解析] 深度兜底: {u[:100]}")
            
    return images


def _extract_images_deep(obj) -> list:
    """从任意嵌套响应中提取图片 URL 或 base64。"""
    images = []
    if isinstance(obj, dict):
        for key in ("url", "image_url", "output_url", "result_url", "content_url", "video_url"):
            val = obj.get(key)
            if isinstance(val, dict):
                val = val.get("url")
            if isinstance(val, str) and val.strip():
                images.append({"type": "url", "data": val.strip()})
        for key in ("b64_json", "base64", "image_base64"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                data = val.split(",", 1)[1] if val.startswith("data:image") and "," in val else val
                images.append({"type": "b64_json", "data": data})
        for val in obj.values():
            if isinstance(val, (dict, list)):
                images.extend(_extract_images_deep(val))
            elif isinstance(val, str) and (val.startswith("http://") or val.startswith("https://") or val.startswith("data:image")):
                if val.startswith("data:image"):
                    images.append({"type": "b64_json", "data": val.split(",", 1)[1] if "," in val else val})
                else:
                    images.append({"type": "url", "data": val})
    elif isinstance(obj, list):
        for item in obj:
            images.extend(_extract_images_deep(item))

    deduped = []
    seen = set()
    for image in images:
        key = (image.get("type"), image.get("data"))
        if key[1] and key not in seen:
            seen.add(key)
            deduped.append(image)
    return deduped


def _parse_images_response(result: dict) -> list:
    """解析 /v1/images/generations 响应，提取图片URL和base64列表。"""
    images = []
    data_items = result.get("data", [])
    if not isinstance(data_items, list):
        data_items = []
    for item in data_items:
        if not isinstance(item, dict):
            continue
        url = item.get("url", "") or item.get("image_url", "") or item.get("output_url", "") or item.get("result_url", "") or item.get("content_url", "")
        if isinstance(url, dict):
            url = url.get("url", "")
        b64 = item.get("b64_json", "") or item.get("base64", "") or item.get("image_base64", "")
        revised_prompt = item.get("revised_prompt", "")
        
        if b64:
            images.append({"type": "b64_json", "data": str(b64)})
            print(f"  [解析] 找到 b64_json ({len(str(b64))//1024}KB)")
        elif url:
            images.append({"type": "url", "data": str(url)})
            print(f"  [解析] 找到 url: {str(url)[:100]}")
        else:
            nested = _extract_images_deep(item)
            if nested:
                images.extend(nested)
                print(f"  [解析] 嵌套字段提取{len(nested)}张")
            elif revised_prompt:
                # 有些API用revised_prompt返回
                print(f"  [解析] 跳过 revised_prompt")
            
    if not images:
        # 兜底：直接在result中找任何图片URL
        import re
        result_str = json.dumps(result, ensure_ascii=False)
        urls = re.findall(r'(?:https?:)?(?://[^\s\"\'<>]+?(?:\.(?:png|jpg|jpeg|webp|gif)[^\s\"\'<>]*)?)', result_str)
        relative_urls = re.findall(r'["\']\s*(/generated/[^\s\"\'<>]+?\.(?:png|jpg|jpeg|webp|gif))["\']', result_str)
        for u in urls:
            images.append({"type": "url", "data": u})
            print(f"  [解析] 兜底找到图片: {u[:100]}")
        for u in relative_urls:
            if not any(img["data"] == u for img in images):
                images.append({"type": "url", "data": u})
                print(f"  [解析] 兜底找到相对URL: {u[:100]}")
                
    return images


def _save_image(image_info: dict, project_path: str, viewer_index: int,
                unique_name: str, generation_round: int, position: int,
                idx: int, n: int, timeout: int, base_url: str = "", api_key: str = "") -> str:
    """保存一张图片到本地，返回文件路径。base_url用于拼接相对路径URL。"""
    suffix = f"_{idx+1}" if n > 1 else ""
    image_name = f"{viewer_index:04d}_{unique_name}_{generation_round}_{position}{suffix}.png"
    image_path = os.path.join(project_path, image_name)
    os.makedirs(project_path, exist_ok=True)
    
    img_type = image_info["type"]
    img_data = image_info["data"]
    
    # URL标准化：相对路径补全base_url，多策略尝试
    def _normalize_url(url):
        if (url.startswith("http://127.0.0.1") or url.startswith("http://localhost") or
                url.startswith("https://127.0.0.1") or url.startswith("https://localhost")):
            match = re.search(r"https?://(?:127\.0\.0\.1|localhost)(?::\d+)?(/.*)$", url)
            if match and base_url:
                root = base_url.rstrip("/")
                if root.endswith("/v1"):
                    root = root[:-3]
                fixed = root + match.group(1)
                print(f"  [URL] localhost地址改写: {url[:80]} -> {fixed[:120]}")
                _debug_log(f"localhost地址改写: {url[:300]} -> {fixed[:300]}")
                return fixed
        if url.startswith("http://") or url.startswith("https://"):
            return url
        if url.startswith("/"):
            if base_url:
                full = base_url.rstrip("/") + url
                print(f"  [URL] 相对路径补全: {url[:60]} -> {full[:120]}")
                return full
            else:
                print(f"  [WARN] 相对URL但无base_url可拼接: {url}")
        # 试试去掉开头的多余路径
        if url.startswith("generated/") and base_url:
            return base_url.rstrip("/") + "/" + url
        print(f"  [WARN] 无法标准化的URL: {url[:80]}")
        return url

    def _should_send_download_auth(url):
        if not api_key or not base_url:
            return False
        try:
            url_host = (urlparse(url).hostname or "").lower()
            base_host = (urlparse(base_url).hostname or "").lower()
        except Exception:
            return False
        if not url_host or not base_host:
            return False
        return url_host == base_host or url_host.endswith("." + base_host)
    
    if img_type == "b64_json":
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(img_data))
        print(f"  图片已保存(base64): {image_path}")
    elif img_type in ("url", "image_url"):
        full_url = _normalize_url(img_data)
        candidate_urls = [full_url]
        if "/generated/" in full_url and base_url:
            root = base_url.rstrip("/")
            if root.endswith("/v1"):
                root = root[:-3]
            generated_path = full_url[full_url.find("/generated/"):]
            candidate_urls.append(root + "/v1" + generated_path)
            candidate_urls.append(root + generated_path)
        
        # 多策略下载
        def _try_download(url, desc=""):
            print(f"  正在下载{desc}: {url[:120]}")
            dl_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "*/*",
            }
            if _should_send_download_auth(url):
                dl_headers["Authorization"] = f"Bearer {api_key}"
            try:
                dl_resp = requests.get(url, headers=dl_headers, timeout=timeout, stream=True)
                if dl_resp.status_code == 200:
                    with open(image_path, "wb") as f:
                        for chunk in dl_resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    try:
                        with Image.open(image_path) as verify_img:
                            verify_img.verify()
                    except Exception as verify_err:
                        head = b""
                        try:
                            with open(image_path, "rb") as f:
                                head = f.read(120)
                            os.remove(image_path)
                        except Exception:
                            pass
                        content_type = dl_resp.headers.get("Content-Type", "")
                        _debug_log(f"下载内容不是有效图片: url={url[:300]} content_type={content_type} head={head[:80]!r} err={verify_err}")
                        print(f"  下载内容不是有效图片: {content_type}")
                        return False
                    print(f"  图片已保存(下载): {image_path}")
                    return True
                else:
                    print(f"  下载失败: HTTP {dl_resp.status_code}")
                    _debug_log(f"下载失败 HTTP {dl_resp.status_code}: {url[:300]} body={dl_resp.text[:300] if hasattr(dl_resp, 'text') else ''}")
                    return False
            except Exception as e:
                print(f"  下载异常: {e}")
                _debug_log(f"下载异常: {e}; url={url[:300]}")
                return False
        
        # 策略1：直接下载
        download_ok = False
        seen_urls = set()
        for candidate in candidate_urls:
            if candidate in seen_urls:
                continue
            seen_urls.add(candidate)
            if _try_download(candidate, ""):
                download_ok = True
                break
        if download_ok:
            pass
        # 策略2：如果失败且是相对路径，尝试用 base_url 拼接
        elif full_url.startswith("/") and base_url:
            alt_url = base_url.rstrip("/") + full_url
            if alt_url != full_url and _try_download(alt_url, "(策略2: base_url拼接)"):
                pass
            else:
                print(f"  所有下载策略均失败: {full_url}")
                return None
        # 策略3：绝对URL但失败
        else:
            print(f"  下载失败且无法尝试备选: {full_url}")
            return None
    elif img_type == "text":
        if img_data.startswith("data:image"):
            header, b64_data = img_data.split(",", 1)
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(b64_data))
            print(f"  图片已保存(data URI): {image_path}")
        elif img_data.startswith("http") or img_data.startswith("/"):
            full_url = _normalize_url(img_data)
            print(f"  正在下载: {full_url[:100]}...")
            dl_resp = requests.get(full_url, timeout=timeout, stream=True)
            if dl_resp.status_code == 200:
                with open(image_path, "wb") as f:
                    for chunk in dl_resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"  图片已保存(下载): {image_path}")
            else:
                print(f"  图片下载失败: HTTP {dl_resp.status_code}, url={full_url}")
                return None
        else:
            print(f"  [结果 {idx+1}] 未知数据格式[{img_data[:50]}...]，跳过")
            return None
    
    return image_path


# ===================== 核心生成 =====================

def generate(context):
    """
    插件主函数：调用 OpenAI 兼容 API 生成图片。
    
    请求 POST {base_url}/v1/images/generations
    兼容 OpenAI 和 NewAPI 图像生成接口。
    """
    print("=" * 80)
    print("[荟聚] 开始生成图片")
    print(f"  [调试] context keys: {list(context.keys())}")
    print(f"  [调试] reference_images: {bool(context.get('reference_images'))}")
    if context.get('reference_images'):
        ri = context['reference_images']
        print(f"  [调试] reference_images type: {type(ri).__name__}")
        if isinstance(ri, dict):
            print(f"  [调试] reference_images keys: {list(ri.keys())}")
            for k in list(ri.keys())[:5]:
                v = ri[k]
                print(f"    [{k}]({type(v).__name__}): {str(v)[:100]}")
        elif isinstance(ri, list):
            print(f"  [调试] reference_images is list, len={len(ri)}")
            for item in ri[:3]:
                print(f"    item: {str(item)[:100]}")
        else:
            print(f"  [调试] reference_images: {str(ri)[:200]}")
    else:
        print(f"  [调试] reference_images 不存在或为空")
    
    try:
        plugin_params = context.get("plugin_params") or get_params()
        
        api_key = str(plugin_params.get("api_key", "")).strip()
        base_url = str(plugin_params.get("base_url", "https://api.openai.com")).strip().rstrip("/")
        model = str(plugin_params.get("model", "dall-e-3")).strip()
        size = str(plugin_params.get("size", "1024x1024")).strip()
        saved_aspect_ratio = plugin_params.get("aspect_ratio") or context.get("aspect_ratio")
        aspect_ratio = _normalize_aspect_ratio_value(
            saved_aspect_ratio or size
        ) or "1:1"
        if not saved_aspect_ratio and _normalize_image_size_value(size):
            aspect_ratio = "1:1"
        image_size = _normalize_image_size_value(
            plugin_params.get("image_size") or context.get("image_size") or size
        ) or "1K"
        if image_size in {"1K", "2K", "4K"}:
            size = _gpt_image_size(aspect_ratio, image_size)
        n = int(plugin_params.get("n", 1))
        quality = str(
            plugin_params.get("gpt_image_quality")
            or plugin_params.get("quality", "standard")
        ).strip()
        style = str(plugin_params.get("style", "vivid")).strip()
        response_format = str(plugin_params.get("response_format", "url")).strip()
        timeout = int(plugin_params.get("timeout", 300))
        
        prompt = context.get("prompt", "")
        face_processing_enabled = _setting_enabled(plugin_params.get("enable_face_processing"), True)
        should_process_character = face_processing_enabled and _is_character_role_request(context, prompt)
        project_path = context.get("project_path", ".")
        unique_name = context.get("unique_name", "test")
        viewer_index = context.get("viewer_index", 1)
        generation_round = context.get("generation_round", 0)
        output_position = context.get("output_position", [])
        batch_num = context.get("batch_num", 1)
        progress_callback = context.get("progress_callback")
        reference_images = context.get("reference_images", {})
        if not reference_images:
            for alias in ("images", "image_paths", "reference_image_paths", "refs", "input_images"):
                if context.get(alias):
                    reference_images = context.get(alias)
                    print(f"  [调试] 从 context.{alias} 读取参考图")
                    break
        
        # ===== 参考图处理（参图生图/img2img） =====
        # 调试：打印原始 reference_images
        if reference_images:
            print(f"  [调试] reference_images 原始类型: {type(reference_images).__name__}")
            if isinstance(reference_images, list):
                print(f"  [调试] reference_images 是列表，长度={len(reference_images)}")
                # 列表转字典
                ref_dict = {}
                for i, item in enumerate(reference_images):
                    if isinstance(item, str):
                        ref_dict[str(i)] = item
                    elif isinstance(item, dict):
                        ref_dict.update(item)
                reference_images = ref_dict
                print(f"  [调试] 列表已转字典: keys={list(reference_images.keys())[:5]}")
            elif isinstance(reference_images, dict):
                print(f"  [调试] reference_images keys: {list(reference_images.keys())}")
                for k, v in reference_images.items():
                    if isinstance(v, dict):
                        print(f"    {k}: dict({len(v)} items), keys={list(v.keys())[:5]}")
                        for sk, sv in list(v.items())[:3]:
                            print(f"      [{sk}] = {str(sv)[:80]}")
                    else:
                        print(f"    {k}: {str(v)[:80]}")
            else:
                print(f"  [调试] reference_images 未知类型，忽略")
                reference_images = {}
        else:
            print(f"  [调试] reference_images 为空！")
        
        # 标准化 reference_images（与视频插件保持一致）
        # 处理各种可能的格式
        if reference_images:
            # 情况1：直接是 {"参考图片MAP": {...}} 格式
            if "参考图片MAP" in reference_images:
                ref_map = reference_images["参考图片MAP"]
                if isinstance(ref_map, dict):
                    reference_images["参考图片MAP"] = {
                        (int(k) if isinstance(k, str) and k.isdigit() else k): v
                        for k, v in ref_map.items()
                    }
            # 情况2：全是数字key → 包装
            elif all(isinstance(k, int) or (isinstance(k, str) and k.isdigit()) for k in reference_images.keys()):
                reference_images = {"参考图片MAP": reference_images.copy()}
                print(f"  [调试] 数字key格式，已包装为参考图片MAP")
            # 情况3：有其他非数字key，检查是否有图片路径
            else:
                # 尝试从任何包含图片路径的key中提取
                found_any = False
                for k, v in list(reference_images.items()):
                    if isinstance(v, str) and os.path.exists(str(v).split("?")[0]):
                        found_any = True
                        print(f"  [调试] 找到图片路径(key={k}): {str(v)[:60]}")
                if not found_any:
                    print(f"  [调试] 未找到有效图片路径，reference_images将不被使用")
                    reference_images = {}
            
            if context.get("first_frame_path"):
                if "参考图片MAP" not in reference_images:
                    reference_images["参考图片MAP"] = {}
                reference_images["首帧"] = context["first_frame_path"]
            if context.get("end_frame_path"):
                if "参考图片MAP" not in reference_images:
                    reference_images["参考图片MAP"] = {}
                reference_images["尾帧"] = context["end_frame_path"]
        
        ref_paths = _collect_reference_images(reference_images) if reference_images else []
        if reference_images:
            print(f"  [调试] 提取到 {len(ref_paths)} 个有效图片路径")
            for rp in ref_paths:
                print(f"    {rp}")
        
        # ===== 清晰的模式日志 =====
        print("=" * 80)
        print("[荟聚] ╔══════════════════════════════════════╗")
        if ref_paths:
            print("[荟聚] ║  🔥 图生图模式 (IMG2IMG)            ║")
            print(f"[荟聚] ║  参考图: {len(ref_paths)} 张                    ║")
            for rp in ref_paths:
                print(f"[荟聚] ║    - {os.path.basename(rp)}")
            print("[荟聚] ║  端点: chat/completions→images/gen ║")
            print("[荟聚] ║  (解析失败自动回退文生图)           ║")
        else:
            print("[荟聚] ║  📝 文生图模式 (TEXT2IMAGE)        ║")
            print("[荟聚] ║  端点: /v1/images/generations      ║")
        print("[荟聚] ╚══════════════════════════════════════╝")
        
        print(f"  Base URL: {base_url}")
        print(f"  模型: {model}")
        print(f"  尺寸: {size}")
        print(f"  数量: {n}")
        print(f"  提示词: {prompt[:100]}...")
        if not face_processing_enabled:
            face_processing_status = "关闭"
        elif should_process_character:
            face_processing_status = "启用（已识别角色图）"
        else:
            face_processing_status = "启用（当前为非角色图，跳过）"
        print(f"  角色图自动人脸处理: {face_processing_status}")
        print("=" * 80)
        
        if not api_key:
            raise Exception("PLUGIN_ERROR:::API Key 未设置，请在插件设置中配置")
        if not base_url:
            raise Exception("PLUGIN_ERROR:::Base URL 未设置")
        if not prompt:
            raise Exception("PLUGIN_ERROR:::提示词为空")
        
        # 图生图优先使用图像生成接口；仅对不支持 images 字段的模型回退 chat/completions。
        # 文生图（无参考图）→ /v1/images/generations。
        is_firefly = _is_firefly_model(model)
        supports_image_refs = _supports_images_generation_refs(model)
        use_chat = bool(ref_paths) and not supports_image_refs
        if ref_paths and supports_image_refs:
            print("  [图生图] 检测到参考图，使用 /v1/images/generations + images 字段")
        
        if use_chat:
            content = [{"type": "text", "text": prompt}]
            total_size = 0
            MAX_PAYLOAD_KB = 102400  # 单次请求上限100MB
            for rp in ref_paths:
                b64 = _read_image_as_base64(rp)
                if b64:
                    img_size = len(b64) // 1024
                    if total_size + img_size > MAX_PAYLOAD_KB:
                        print(f"  [警告] payload过大，跳过后续参考图")
                        break
                    content.append({"type": "image_url", "image_url": {"url": b64}})
                    total_size += img_size
                    print(f"  参考图已编码: {os.path.basename(rp)} ({img_size}KB)")
        
        result_paths = []
        last_error = None
        
        for i in range(batch_num):
            try:
                position = output_position[i] if i < len(output_position) else i
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                }
                
                if use_chat:
                    # 方案A：图生图 - chat/completions
                    endpoint = f"{base_url}/v1/chat/completions"
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": 4096,
                    }
                    _apply_weigrok_resolution_payload(payload, aspect_ratio, image_size)
                    gpt_image_quality = _gpt_image_quality_for_request(model, image_size, quality)
                    if gpt_image_quality:
                        payload["gpt_image_quality"] = gpt_image_quality
                        payload.setdefault("extra_body", {})["gpt_image_quality"] = gpt_image_quality
                    print(f"  [图生图] /v1/chat/completions，{len(content)-1}张参考图")
                else:
                    # 方案B：文生图 - images/generations
                    endpoint = f"{base_url}/v1/images/generations"
                    payload = {
                        "model": model, "prompt": prompt,
                        "n": n, "size": size, "response_format": response_format,
                    }
                    if ref_paths and supports_image_refs:
                        payload["images"] = [_read_image_as_base64(rp) for rp in ref_paths if rp]
                        payload["images"] = [img for img in payload["images"] if img]
                        print(f"  [图生图] images 字段参考图数量: {len(payload['images'])}")
                    _apply_weigrok_resolution_payload(payload, aspect_ratio, image_size)
                    gpt_image_quality = _gpt_image_quality_for_request(model, image_size, quality)
                    if gpt_image_quality:
                        payload["gpt_image_quality"] = gpt_image_quality
                        payload.setdefault("extra_body", {})["gpt_image_quality"] = gpt_image_quality
                    if "gpt-image" in model.lower() and "gpt-image-2" not in model.lower() and not _is_firefly_model(model):
                        payload["quality"] = {"standard":"medium","hd":"high","low":"low","medium":"medium","high":"high"}.get(quality,"medium")
                    elif "dall-e-3" in model.lower():
                        payload["quality"] = quality if quality in ("standard","hd") else "standard"
                        payload["style"] = style if style in ("vivid","natural") else "vivid"
                    print(f"  [文生图] /v1/images/generations")
                
                if progress_callback:
                    progress_callback(f"生成中 ({i+1}/{batch_num})...")
                
                print(f"  发送请求 [{i+1}/{batch_num}]: {endpoint}")
                response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
                
                if response.status_code != 200:
                    detail = ""
                    try: detail = response.json()
                    except: detail = response.text[:500]
                    raise Exception(f"PLUGIN_ERROR:::API错误{response.status_code}: {detail}")
                
                result = response.json()
                print(f"  API响应(前300字): {json.dumps(result, ensure_ascii=False)[:300]}")
                print(f"  API响应(完整): {json.dumps(result, ensure_ascii=False)[:1500]}")
                _debug_log(f"请求endpoint={endpoint}")
                _debug_log(f"请求payload={json.dumps(payload, ensure_ascii=False)[:3000]}")
                _debug_log(f"API响应={json.dumps(result, ensure_ascii=False)[:5000]}")
                
                # 解析
                images = _parse_chat_response(result, base_url) if use_chat else _parse_images_response(result)
                
                # 如果用chat/completions解析失败，回退到文生图
                if not images and use_chat:
                    print(f"  [回退] chat/completions解析失败，改用 images/generations")
                    fallback_endpoint = f"{base_url}/v1/images/generations"
                    fallback_payload = {
                        "model": model, "prompt": prompt,
                        "n": n, "size": size, "response_format": response_format,
                    }
                    if ref_paths and supports_image_refs:
                        fallback_payload["images"] = [_read_image_as_base64(rp) for rp in ref_paths if rp]
                        fallback_payload["images"] = [img for img in fallback_payload["images"] if img]
                    _apply_weigrok_resolution_payload(fallback_payload, aspect_ratio, image_size)
                    gpt_image_quality = _gpt_image_quality_for_request(model, image_size, quality)
                    if gpt_image_quality:
                        fallback_payload["gpt_image_quality"] = gpt_image_quality
                        fallback_payload.setdefault("extra_body", {})["gpt_image_quality"] = gpt_image_quality
                    if "gpt-image" in model.lower() and "gpt-image-2" not in model.lower() and not _is_firefly_model(model):
                        fallback_payload["quality"] = {"standard":"medium","hd":"high","low":"low","medium":"medium","high":"high"}.get(quality,"medium")
                    elif "dall-e-3" in model.lower():
                        fallback_payload["quality"] = quality if quality in ("standard","hd") else "standard"
                        fallback_payload["style"] = style if style in ("vivid","natural") else "vivid"
                    
                    fb_resp = requests.post(fallback_endpoint, headers=headers, json=fallback_payload, timeout=timeout)
                    if fb_resp.status_code != 200:
                        detail = ""
                        try: detail = fb_resp.json()
                        except: detail = fb_resp.text[:500]
                        raise Exception(f"PLUGIN_ERROR:::回退也失败{fb_resp.status_code}: {detail}")
                    
                    result = fb_resp.json()
                    print(f"  回退API响应: {json.dumps(result, ensure_ascii=False)[:300]}")
                    images = _parse_images_response(result)
                
                if not images:
                    print(f"  [荟聚] 解析失败！keys: {list(result.keys())}")
                    print(f"  [荟聚] 完整: {json.dumps(result, ensure_ascii=False)[:1000]}")
                    raise Exception("PLUGIN_ERROR:::API未返回有效图片数据")
                
                for idx, img_info in enumerate(images):
                    saved = _save_image(
                        img_info, project_path, viewer_index,
                        unique_name, generation_round, position, idx, len(images), timeout,
                        base_url=base_url, api_key=api_key
                    )
                    if saved:
                        # Keep the output path unchanged: 字字's preview, project data and any
                        # later reference-image upload therefore all use this processed file.
                        if should_process_character:
                            _process_character_role_image(saved)
                        result_paths.append(saved)
                    else:
                        _debug_log(f"图片保存失败: {json.dumps(img_info, ensure_ascii=False)[:1000]}")
                
            except Exception as e:
                error_msg = str(e)
                last_error = error_msg
                print(f"  生成第 {i+1} 轮失败: {error_msg}")
                if error_msg.startswith("PLUGIN_ERROR:::"):
                    raise
                traceback.print_exc()
        
        if progress_callback:
            progress_callback("完成", 100)
        if not result_paths and last_error:
            raise Exception(f"PLUGIN_ERROR:::{last_error}")
        if not result_paths:
            raise Exception(f"PLUGIN_ERROR:::插件未保存任何图片文件，请查看日志: {plugin_dir / 'logs'}")
        
        mode_icon = "🔥图生图" if ref_paths else "📝文生图"
        print(f"[荟聚] {mode_icon} 完成，共 {len(result_paths)} 个文件")
        for p in result_paths:
            print(f"[荟聚]   → {os.path.basename(p)}")
        print("=" * 80)
        return result_paths
    
    except Exception as e:
        error_msg = str(e)
        print(f"[荟聚] 生成出错: {error_msg}")
        if error_msg.startswith("PLUGIN_ERROR:::"):
            raise
        traceback.print_exc()
        raise Exception(f"PLUGIN_ERROR:::{error_msg}")


# ===================== 插件接口 =====================

def get_info():
    """返回插件信息。"""
    return {
        "name": "荟聚",
        "description": (
            "兼容 OpenAI API 的图像生成插件\n"
            "支持自定义 API 地址和密钥\n"
            "自动从 /v1/models 获取可用模型列表\n"
            "支持参图生图（img2img）：单张或多张参考图\n"
            "兼容 OpenAI 格式的 API 服务（通过荟聚中转 Adobe2api）"
        ),
        "version": _PLUGIN_VERSION,
        "author": "",
    }


def get_params():
    """获取当前参数，从 config.json 实时读取。"""
    params = _DEFAULT_PARAMS.copy()
    params.update(load_plugin_config(_PLUGIN_FILE))
    return params


def handle_action(action, data=None):
    """
    处理前端发来的自定义动作。
    
    支持的动作:
      - fetch_models: 根据 base_url + api_key 从 /v1/models 获取可用模型列表
      - get_size_options: 根据当前模型返回可用尺寸列表
    """
    if data is None:
        data = {}
    
    if action == "fetch_models":
        base_url = str(data.get("base_url", "")).strip().rstrip("/")
        api_key = str(data.get("api_key", "")).strip()
        timeout = int(data.get("timeout", 15))
        
        if not base_url:
            return {"ok": False, "error": "Base URL 不能为空"}
        
        print(f"[荟聚] 正在从 {base_url}/v1/models 获取模型列表...")
        result = _fetch_models_from_api(base_url, api_key, timeout)
        
        if result.get("ok"):
            models = result.get("models", [])
            default_model = result.get("default_model", models[0] if models else "")
            print(f"[荟聚] 获取到 {len(models)} 个模型")
            
            # 同时保存到 config.json，方便 UI 通过 getParams() 获取
            try:
                update_plugin_params(_PLUGIN_FILE, {
                    "model_list": json.dumps(models, ensure_ascii=False),
                    "model_list_updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_list_default": default_model,
                })
                print(f"[荟聚] 模型列表已保存到配置")
            except Exception as save_err:
                print(f"[荟聚] 保存模型列表到配置失败: {save_err}")
            
            return {
                "ok": True,
                "models": models,
                "default_model": default_model,
            }
        else:
            print(f"[荟聚] 获取模型列表失败: {result.get('error')}")
            return {"ok": False, "error": result.get("error", "未知错误")}
    
    elif action == "check_update":
        result = _get_latest_release(_UPDATE_REPO, int(data.get("timeout", 20) or 20))
        if result.get("ok") and not result.get("has_update"):
            result["message"] = "已经是最新版本"
        return result

    elif action == "apply_update":
        asset_name = str(data.get("asset_name") or get_params().get("update_asset_name") or "").strip()
        return _apply_github_update(_UPDATE_REPO, asset_name)

    elif action == "get_size_options":
        model = str(data.get("model", _get_model_from_config())).strip()
        sizes = _get_size_options_for_model(model)
        return {"ok": True, "sizes": sizes, "model": model}
    
    else:
        return {"ok": False, "error": f"未知动作: {action}"}


# ===================== 初始化 =====================

print(f"[荟聚] 插件已加载 (v{_PLUGIN_VERSION})")
