"""
Batch Face Swapper — Streamlit web app.

The studio template is fixed. Users only upload source photos (or a zip of
photos); each face is aligned to the template automatically.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
BASE_IMAGE_PATH = ROOT / "base_image.jpg"
MIN_FACE_PX = 16
MASK_FEATHER_PX = 7
HULL_DILATE_RATIO = 0.06

FACE_LANDMARKER_MODEL = ROOT / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

ALIGN_LANDMARK_IDS = (
    1,
    4,
    5,
    6,
    10,
    13,
    14,
    33,
    61,
    78,
    133,
    152,
    234,
    263,
    291,
    308,
    362,
    454,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff", "zip"]


# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
def ensure_face_landmarker_model() -> Path:
    if FACE_LANDMARKER_MODEL.is_file() and FACE_LANDMARKER_MODEL.stat().st_size > 0:
        return FACE_LANDMARKER_MODEL
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_MODEL)
    if not FACE_LANDMARKER_MODEL.is_file() or FACE_LANDMARKER_MODEL.stat().st_size == 0:
        raise RuntimeError("Face Landmarker model download was empty.")
    return FACE_LANDMARKER_MODEL


@st.cache_resource(show_spinner="Loading face detector…")
def get_landmarker():
    model_path = ensure_face_landmarker_model()
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=8,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def _landmarks_to_pixels(face_landmarks, width: int, height: int) -> np.ndarray:
    points = getattr(face_landmarks, "landmark", face_landmarks)
    pts = np.array(
        [(int(round(lm.x * width)), int(round(lm.y * height))) for lm in points],
        dtype=np.int32,
    )
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    return pts


def _bbox_area(pts: np.ndarray) -> int:
    _x, _y, bw, bh = cv2.boundingRect(pts)
    return int(bw) * int(bh)


def detect_largest_landmarks(image_bgr: np.ndarray, landmarker) -> np.ndarray | None:
    h, w = image_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect(mp_image)
    face_lists = getattr(results, "face_landmarks", None) or getattr(
        results, "multi_face_landmarks", None
    )
    if not face_lists:
        return None
    faces = [_landmarks_to_pixels(face, w, h) for face in face_lists]
    largest = max(faces, key=_bbox_area)
    if _bbox_area(largest) <= 0:
        return None
    return largest


def hull_mask_from_landmarks(
    landmarks: np.ndarray, shape_hw: tuple[int, int], dilate: bool = True
) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate:
        _x, _y, bw, bh = cv2.boundingRect(hull)
        k = max(3, int(HULL_DILATE_RATIO * max(bw, bh))) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)
    feather = MASK_FEATHER_PX | 1
    return cv2.GaussianBlur(mask, (feather, feather), 0)


# ---------------------------------------------------------------------------
# Alignment, lighting, clone
# ---------------------------------------------------------------------------
def _alignment_points(landmarks: np.ndarray) -> np.ndarray:
    n = len(landmarks)
    ids = [i for i in ALIGN_LANDMARK_IDS if i < n]
    if len(ids) >= 3:
        return landmarks[ids].astype(np.float32)
    return landmarks.astype(np.float32)


def similarity_from_bboxes(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    sx, sy, sw, sh = cv2.boundingRect(src_pts)
    dx, dy, dw, dh = cv2.boundingRect(dst_pts)
    scale = min(dw / max(sw, 1), dh / max(sh, 1))
    src_cx, src_cy = sx + sw * 0.5, sy + sh * 0.5
    dst_cx, dst_cy = dx + dw * 0.5, dy + dh * 0.5
    return np.array(
        [[scale, 0.0, dst_cx - scale * src_cx], [0.0, scale, dst_cy - scale * src_cy]],
        dtype=np.float32,
    )


def estimate_face_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    src_a, dst_a = _alignment_points(src_pts), _alignment_points(dst_pts)
    count = min(len(src_a), len(dst_a))
    if count >= 3:
        matrix, _ = cv2.estimateAffinePartial2D(
            src_a[:count], dst_a[:count], method=cv2.RANSAC, ransacReprojThreshold=8.0
        )
        if matrix is not None:
            return matrix.astype(np.float32)
    return similarity_from_bboxes(src_pts, dst_pts)


def align_source_to_base(
    source_bgr: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    dst_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the source face onto the template face (size, pose, and center)."""
    dh, dw = dst_shape
    matrix = estimate_face_transform(src_pts, dst_pts)
    warped = cv2.warpAffine(
        source_bgr, matrix, (dw, dh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    dest_mask = hull_mask_from_landmarks(dst_pts, (dh, dw), dilate=True)
    return warped, dest_mask


def match_face_lighting(
    src_bgr: np.ndarray, dst_bgr: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    apply = mask > 16
    if int(np.count_nonzero(apply)) < 32:
        return src_bgr

    k = max(3, int(0.08 * min(mask.shape[:2])) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inner = cv2.erode((mask > 16).astype(np.uint8) * 255, kernel)
    stats = inner > 0
    if int(np.count_nonzero(stats)) < 64:
        stats = apply

    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out_lab = src_lab.copy()

    for channel in range(3):
        src_vals = src_lab[:, :, channel][stats]
        dst_vals = dst_lab[:, :, channel][stats]
        src_mean, dst_mean = float(src_vals.mean()), float(dst_vals.mean())
        src_std, dst_std = float(src_vals.std()), float(dst_vals.std())
        scale = float(np.clip(dst_std / (src_std + 1e-6), 0.35, 2.8))
        transferred = (src_lab[:, :, channel] - src_mean) * scale + dst_mean
        out_lab[:, :, channel] = np.where(apply, transferred, src_lab[:, :, channel])

    src_l, dst_l = out_lab[:, :, 0][stats], dst_lab[:, :, 0][stats]
    l_gain = float(np.clip(float(dst_l.mean()) / (float(src_l.mean()) + 1e-6), 0.55, 1.85))
    out_lab[:, :, 0] = np.where(apply, out_lab[:, :, 0] * l_gain, out_lab[:, :, 0])

    matched = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    blended = matched.astype(np.float32) * alpha + src_bgr.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def crop_to_clone_bounds(
    face_bgr: np.ndarray,
    mask: np.ndarray,
    center: tuple[int, int],
    dst_w: int,
    dst_h: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    fh, fw = face_bgr.shape[:2]
    cx, cy = center
    x1, y1 = cx - fw // 2, cy - fh // 2
    x2, y2 = x1 + fw, y1 + fh
    crop_x0, crop_y0 = max(0, -x1), max(0, -y1)
    crop_x1, crop_y1 = fw - max(0, x2 - dst_w), fh - max(0, y2 - dst_h)
    if crop_x1 - crop_x0 < MIN_FACE_PX or crop_y1 - crop_y0 < MIN_FACE_PX:
        return None
    face_c = face_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
    mask_c = mask[crop_y0:crop_y1, crop_x0:crop_x1]
    ch, cw = face_c.shape[:2]
    new_x1, new_y1 = x1 + crop_x0, y1 + crop_y0
    new_cx, new_cy = new_x1 + cw // 2, new_y1 + ch // 2
    if new_x1 < 0 or new_y1 < 0 or new_x1 + cw > dst_w or new_y1 + ch > dst_h:
        new_cx = int(np.clip(new_cx, cw // 2, dst_w - (cw - cw // 2)))
        new_cy = int(np.clip(new_cy, ch // 2, dst_h - (ch - ch // 2)))
    return face_c, mask_c, (int(new_cx), int(new_cy))


def overlay_fits(face_bgr: np.ndarray, center: tuple[int, int], dst_w: int, dst_h: int) -> bool:
    fh, fw = face_bgr.shape[:2]
    cx, cy = center
    x1, y1 = cx - fw // 2, cy - fh // 2
    return x1 >= 0 and y1 >= 0 and x1 + fw <= dst_w and y1 + fh <= dst_h


def alpha_blend(
    dst: np.ndarray, face_bgr: np.ndarray, mask: np.ndarray, center: tuple[int, int]
) -> np.ndarray:
    out = dst.copy()
    fh, fw = face_bgr.shape[:2]
    cx, cy = center
    x1, y1 = cx - fw // 2, cy - fh // 2
    x2, y2 = x1 + fw, y1 + fh
    dh, dw = out.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > dw or y2 > dh:
        fitted = crop_to_clone_bounds(face_bgr, mask, center, dw, dh)
        if fitted is None:
            raise RuntimeError("Face region is outside the base image.")
        face_bgr, mask, center = fitted
        fh, fw = face_bgr.shape[:2]
        cx, cy = center
        x1, y1 = cx - fw // 2, cy - fh // 2
        x2, y2 = x1 + fw, y1 + fh
    roi = out[y1:y2, x1:x2]
    alpha = np.expand_dims(mask.astype(np.float32) / 255.0, axis=2)
    blended = face_bgr.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
    out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def composite_face(
    base_bgr: np.ndarray,
    face_bgr: np.ndarray,
    mask: np.ndarray,
    center: tuple[int, int],
) -> np.ndarray:
    dst = base_bgr.copy()
    dh, dw = dst.shape[:2]
    prepared = crop_to_clone_bounds(face_bgr, mask, center, dw, dh)
    if prepared is None:
        raise RuntimeError("Cannot place the face without leaving the image.")
    face_bgr, mask, center = prepared
    if not overlay_fits(face_bgr, center, dw, dh):
        return alpha_blend(dst, face_bgr, mask, center)
    mask_clone = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.ndim == 3 else mask
    try:
        if cv2.countNonZero(mask_clone) == 0:
            raise RuntimeError("Empty face mask.")
        return cv2.seamlessClone(face_bgr, dst, mask_clone, center, cv2.NORMAL_CLONE)
    except Exception:
        return alpha_blend(dst, face_bgr, mask_clone, center)


def patch_from_aligned(
    warped_bgr: np.ndarray, dest_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    ys, xs = np.where(dest_mask > 16)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad = max(2, int(0.04 * max(x1 - x0, y1 - y0)))
    h, w = warped_bgr.shape[:2]
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    face = warped_bgr[y0:y1, x0:x1].copy()
    mask = dest_mask[y0:y1, x0:x1].copy()
    if face.shape[0] < MIN_FACE_PX or face.shape[1] < MIN_FACE_PX:
        return None
    return face, mask, (x0 + face.shape[1] // 2, y0 + face.shape[0] // 2)


# ---------------------------------------------------------------------------
# Template + batch
# ---------------------------------------------------------------------------
def decode_bytes(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_jpeg(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("Failed to encode JPEG.")
    return buf.tobytes()


@st.cache_resource(show_spinner="Preparing studio template…")
def load_template() -> tuple[np.ndarray, np.ndarray]:
    """Load the fixed template and detect the face that every swap targets."""
    if not BASE_IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {BASE_IMAGE_PATH.name}. Place the studio template next to app.py."
        )
    image = cv2.imread(str(BASE_IMAGE_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not read base_image.jpg.")
    dest_pts = detect_largest_landmarks(image, get_landmarker())
    if dest_pts is None:
        raise RuntimeError("No face detected on the studio template.")
    return image, dest_pts


def collect_uploads(uploaded_files) -> tuple[list[tuple[str, np.ndarray]], list[str]]:
    """Turn loose images and zip archives into (name, bgr) pairs."""
    sources: list[tuple[str, np.ndarray]] = []
    warnings: list[str] = []
    used_stems: set[str] = set()

    def add_image(name: str, data: bytes) -> None:
        img = decode_bytes(data)
        if img is None:
            warnings.append(f"Skipping {name}: Could not read image")
            return
        stem = Path(name).stem or "face"
        unique = stem
        n = 2
        while unique in used_stems:
            unique = f"{stem}_{n}"
            n += 1
        used_stems.add(unique)
        sources.append((f"{unique}{Path(name).suffix or '.jpg'}", img))

    for uploaded in uploaded_files:
        name = uploaded.name
        data = uploaded.getvalue()
        suffix = Path(name).suffix.lower()
        if suffix == ".zip":
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    for info in archive.infolist():
                        if info.is_dir():
                            continue
                        inner = Path(info.filename)
                        if inner.name.startswith(".") or "__MACOSX" in inner.parts:
                            continue
                        if inner.suffix.lower() not in IMAGE_EXTS:
                            continue
                        add_image(inner.name, archive.read(info))
            except zipfile.BadZipFile:
                warnings.append(f"Skipping {name}: Not a valid zip file")
            continue
        if suffix not in IMAGE_EXTS:
            warnings.append(f"Skipping {name}: Unsupported file type")
            continue
        add_image(name, data)

    return sources, warnings


def process_sources(
    base_bgr: np.ndarray,
    dest_pts: np.ndarray,
    sources: list[tuple[str, np.ndarray]],
) -> tuple[list[dict], list[str]]:
    landmarker = get_landmarker()
    dh, dw = base_bgr.shape[:2]
    results: list[dict] = []
    warnings: list[str] = []

    for name, source in sources:
        src_pts = detect_largest_landmarks(source, landmarker)
        if src_pts is None:
            warnings.append(f"Skipping {name}: No face detected")
            continue
        try:
            warped, dest_mask = align_source_to_base(source, src_pts, dest_pts, (dh, dw))
            warped = match_face_lighting(warped, base_bgr, dest_mask)
            patched = patch_from_aligned(warped, dest_mask)
            if patched is None:
                warnings.append(f"Skipping {name}: Aligned face region was empty")
                continue
            face_patch, mask_patch, center = patched
            merged = composite_face(base_bgr, face_patch, mask_patch, center)
        except Exception as exc:
            warnings.append(f"Skipping {name}: Composite failed ({exc})")
            continue

        out_name = f"merged_{Path(name).stem}.jpg"
        rgb = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)
        results.append(
            {"name": out_name, "image_rgb": rgb, "jpeg_bytes": encode_jpeg(merged)}
        )
    return results, warnings


def zip_results(results: list[dict]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in results:
            archive.writestr(item["name"], item["jpeg_bytes"])
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Batch Face Swapper", layout="wide")
    st.title("Batch Face Swapper")
    st.markdown(
        "Every photo is swapped onto the same studio template automatically. "
        "Upload **one image**, **many images**, or a **.zip folder** — then download the results."
    )

    if "results" not in st.session_state:
        st.session_state.results = []
        st.session_state.warnings = []
        st.session_state.last_job = None

    try:
        base_bgr, dest_pts = load_template()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    preview, uploader = st.columns([1, 1.2], gap="large")
    with preview:
        st.caption("Fixed template (face detected automatically)")
        st.image(
            cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB),
            caption="Studio template",
            use_container_width=True,
        )
    with uploader:
        st.subheader("Upload faces")
        uploaded_files = st.file_uploader(
            "Photos or a zip of photos",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            help="Select one file, many files, or a .zip of a folder. The largest face in each photo is used.",
        )

    if not uploaded_files:
        st.info("Drop photos here. Processing starts as soon as they upload.")
        st.stop()

    job_key = tuple((f.name, f.size) for f in uploaded_files)
    if st.session_state.last_job != job_key:
        sources, decode_warnings = collect_uploads(uploaded_files)
        with st.spinner(f"Swapping {len(sources)} photo(s) onto the template…"):
            results, warnings = process_sources(base_bgr, dest_pts, sources)
        st.session_state.results = results
        st.session_state.warnings = decode_warnings + warnings
        st.session_state.last_job = job_key

    for message in st.session_state.warnings:
        st.warning(message)

    st.subheader("Results")
    if not st.session_state.results:
        st.error("No faces were composited. Check the warnings above.")
        return

    st.download_button(
        label=f"Download all ({len(st.session_state.results)}) as ZIP",
        data=zip_results(st.session_state.results),
        file_name="face_swaps.zip",
        mime="application/zip",
        type="primary",
    )

    cols = st.columns(min(3, len(st.session_state.results)))
    for i, item in enumerate(st.session_state.results):
        with cols[i % len(cols)]:
            st.image(item["image_rgb"], caption=item["name"], use_container_width=True)
            st.download_button(
                label=f"Download {item['name']}",
                data=item["jpeg_bytes"],
                file_name=item["name"],
                mime="image/jpeg",
                key=f"dl_{item['name']}_{i}",
            )


if __name__ == "__main__":
    main()
