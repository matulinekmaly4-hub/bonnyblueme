"""
Batch Face Swapper
------------------
Detects the face on base_image.jpg, then warps each source face onto that
same location/size/pose using MediaPipe landmarks and seamless cloning.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
BASE_IMAGE_PATH = ROOT / "base_image.jpg"
SOURCES_DIR = ROOT / "sources"
OUTPUTS_DIR = ROOT / "outputs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MIN_FACE_PX = 16
MASK_FEATHER_PX = 7  # Soften hull edges for smoother blending
HULL_DILATE_RATIO = 0.06  # Grow dest hull slightly so cheeks/forehead are covered

# MediaPipe 1.0 removed mp.solutions.face_mesh. Face Landmarker (Tasks API)
# needs this .task file; it is downloaded once into the project root.
FACE_LANDMARKER_MODEL = ROOT / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Stable MediaPipe Face Mesh indices for pose alignment (eyes, nose, mouth, chin).
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


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def ensure_directories() -> None:
    """Create sources/ and outputs/ if they are missing."""
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_base_image() -> np.ndarray:
    """Load base_image.jpg or abort with a clear error."""
    if not BASE_IMAGE_PATH.is_file():
        print(
            f"ERROR: Target template '{BASE_IMAGE_PATH.name}' was not found in "
            f"{ROOT}. Place a file named base_image.jpg in the project root "
            "and run the script again.",
            file=sys.stderr,
        )
        sys.exit(1)

    image = cv2.imread(str(BASE_IMAGE_PATH), cv2.IMREAD_COLOR)
    if image is None:
        print(
            f"ERROR: Failed to read '{BASE_IMAGE_PATH}'. "
            "The file may be corrupt or not a valid image.",
            file=sys.stderr,
        )
        sys.exit(1)
    return image


def list_source_images() -> list[Path]:
    """Return sorted image files in sources/ (non-recursive)."""
    files = [
        p
        for p in SOURCES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


# ---------------------------------------------------------------------------
# Face detection (MediaPipe Face Landmarker)
# ---------------------------------------------------------------------------
def ensure_face_landmarker_model() -> Path:
    """Download the Face Landmarker model if it is not already on disk."""
    if FACE_LANDMARKER_MODEL.is_file() and FACE_LANDMARKER_MODEL.stat().st_size > 0:
        return FACE_LANDMARKER_MODEL

    print(f"Downloading Face Landmarker model to {FACE_LANDMARKER_MODEL.name}...")
    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_MODEL)
    except Exception as exc:
        if FACE_LANDMARKER_MODEL.exists():
            FACE_LANDMARKER_MODEL.unlink(missing_ok=True)
        print(
            "ERROR: Could not download the MediaPipe Face Landmarker model.\n"
            f"  {exc}\n"
            f"  Save it manually as {FACE_LANDMARKER_MODEL} from:\n"
            f"  {FACE_LANDMARKER_URL}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not FACE_LANDMARKER_MODEL.is_file() or FACE_LANDMARKER_MODEL.stat().st_size == 0:
        print("ERROR: Face Landmarker model download was empty.", file=sys.stderr)
        sys.exit(1)
    return FACE_LANDMARKER_MODEL


def create_face_landmarker():
    """Build a Face Landmarker in IMAGE mode (MediaPipe 1.0 Tasks API)."""
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
    """Convert normalized landmarks to integer pixel coordinates."""
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
    """Return pixel landmarks of the largest face, or None if none were found."""
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
    """Face-shaped (convex hull) mask, optionally dilated then feathered."""
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
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    return mask


# ---------------------------------------------------------------------------
# Landmark alignment
# ---------------------------------------------------------------------------
def _alignment_points(landmarks: np.ndarray) -> np.ndarray:
    """Subset of landmarks used to estimate scale/rotation/translation."""
    n = len(landmarks)
    ids = [i for i in ALIGN_LANDMARK_IDS if i < n]
    if len(ids) >= 3:
        return landmarks[ids].astype(np.float32)
    return landmarks.astype(np.float32)


def similarity_from_bboxes(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Fallback: match bounding-box size and center (no rotation)."""
    sx, sy, sw, sh = cv2.boundingRect(src_pts)
    dx, dy, dw, dh = cv2.boundingRect(dst_pts)
    scale = min(dw / max(sw, 1), dh / max(sh, 1))
    src_cx = sx + sw * 0.5
    src_cy = sy + sh * 0.5
    dst_cx = dx + dw * 0.5
    dst_cy = dy + dh * 0.5
    return np.array(
        [[scale, 0.0, dst_cx - scale * src_cx], [0.0, scale, dst_cy - scale * src_cy]],
        dtype=np.float32,
    )


def estimate_face_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Similarity transform mapping source landmarks onto the base-image face.
    Eyes/nose/mouth stay aligned; scale matches the template face, not a
    fixed fraction of the canvas.
    """
    src_a = _alignment_points(src_pts)
    dst_a = _alignment_points(dst_pts)
    count = min(len(src_a), len(dst_a))
    if count >= 3:
        matrix, _inliers = cv2.estimateAffinePartial2D(
            src_a[:count],
            dst_a[:count],
            method=cv2.RANSAC,
            ransacReprojThreshold=8.0,
        )
        if matrix is not None:
            return matrix.astype(np.float32)
    return similarity_from_bboxes(src_pts, dst_pts)


def align_source_to_base(
    source_bgr: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, dst_shape: tuple[int, int]
) -> np.ndarray:
    """Warp the full source photo so its face sits on the template face."""
    dh, dw = dst_shape
    matrix = estimate_face_transform(src_pts, dst_pts)
    return cv2.warpAffine(
        source_bgr,
        matrix,
        (dw, dh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def match_face_lighting(
    src_bgr: np.ndarray, dst_bgr: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    Match the source face to the template's lighting and color.

    Uses Reinhard transfer in LAB (brightness L + chroma a/b) so a warm
    indoor source picks up the template's cooler, brighter studio look.
    Statistics are taken from the inner face so hair around the hull
    does not dominate the match.
    """
    if src_bgr.shape[:2] != dst_bgr.shape[:2]:
        raise ValueError("Source and template must be the same size after alignment.")

    apply = mask > 16
    if int(np.count_nonzero(apply)) < 32:
        return src_bgr

    # Shrink the mask so means/stds come from skin, not hair/background.
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
        src_mean = float(src_vals.mean())
        dst_mean = float(dst_vals.mean())
        src_std = float(src_vals.std())
        dst_std = float(dst_vals.std())
        scale = dst_std / (src_std + 1e-6)
        scale = float(np.clip(scale, 0.35, 2.8))
        transferred = (src_lab[:, :, channel] - src_mean) * scale + dst_mean
        out_lab[:, :, channel] = np.where(apply, transferred, src_lab[:, :, channel])

    # Extra luminance lock: after chroma transfer, snap mean L to the template.
    src_l = out_lab[:, :, 0][stats]
    dst_l = dst_lab[:, :, 0][stats]
    l_gain = float(dst_l.mean() / (float(src_l.mean()) + 1e-6))
    l_gain = float(np.clip(l_gain, 0.55, 1.85))
    out_lab[:, :, 0] = np.where(apply, out_lab[:, :, 0] * l_gain, out_lab[:, :, 0])

    out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    matched = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

    # Keep correction inside the face; leave warped background untouched.
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    blended = matched.astype(np.float32) * alpha + src_bgr.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Bounds-safe compositing
# ---------------------------------------------------------------------------
def crop_to_clone_bounds(
    face_bgr: np.ndarray,
    mask: np.ndarray,
    center: tuple[int, int],
    dst_w: int,
    dst_h: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    """Crop so seamlessClone's source rectangle stays inside the destination."""
    fh, fw = face_bgr.shape[:2]
    cx, cy = center
    x1 = cx - fw // 2
    y1 = cy - fh // 2
    x2 = x1 + fw
    y2 = y1 + fh

    crop_x0 = max(0, -x1)
    crop_y0 = max(0, -y1)
    crop_x1 = fw - max(0, x2 - dst_w)
    crop_y1 = fh - max(0, y2 - dst_h)

    if crop_x1 - crop_x0 < MIN_FACE_PX or crop_y1 - crop_y0 < MIN_FACE_PX:
        return None

    face_c = face_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
    mask_c = mask[crop_y0:crop_y1, crop_x0:crop_x1]
    ch, cw = face_c.shape[:2]

    new_x1 = x1 + crop_x0
    new_y1 = y1 + crop_y0
    new_cx = new_x1 + cw // 2
    new_cy = new_y1 + ch // 2

    if new_x1 < 0 or new_y1 < 0 or new_x1 + cw > dst_w or new_y1 + ch > dst_h:
        new_cx = int(np.clip(new_cx, cw // 2, dst_w - (cw - cw // 2)))
        new_cy = int(np.clip(new_cy, ch // 2, dst_h - (ch - ch // 2)))

    return face_c, mask_c, (int(new_cx), int(new_cy))


def overlay_fits(face_bgr: np.ndarray, center: tuple[int, int], dst_w: int, dst_h: int) -> bool:
    fh, fw = face_bgr.shape[:2]
    cx, cy = center
    x1 = cx - fw // 2
    y1 = cy - fh // 2
    return x1 >= 0 and y1 >= 0 and x1 + fw <= dst_w and y1 + fh <= dst_h


def alpha_blend(
    dst: np.ndarray,
    face_bgr: np.ndarray,
    mask: np.ndarray,
    center: tuple[int, int],
) -> np.ndarray:
    """Fallback: masked alpha blend of the face onto dst around `center`."""
    out = dst.copy()
    fh, fw = face_bgr.shape[:2]
    cx, cy = center
    x1 = cx - fw // 2
    y1 = cy - fh // 2
    x2 = x1 + fw
    y2 = y1 + fh

    dh, dw = out.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > dw or y2 > dh:
        fitted = crop_to_clone_bounds(face_bgr, mask, center, dw, dh)
        if fitted is None:
            raise RuntimeError("Face region is outside the base image.")
        face_bgr, mask, center = fitted
        fh, fw = face_bgr.shape[:2]
        cx, cy = center
        x1 = cx - fw // 2
        y1 = cy - fh // 2
        x2 = x1 + fw
        y2 = y1 + fh

    roi = out[y1:y2, x1:x2]
    alpha = mask.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    blended = face_bgr.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
    out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def composite_face(
    base_bgr: np.ndarray,
    face_bgr: np.ndarray,
    mask: np.ndarray,
    center: tuple[int, int],
) -> np.ndarray:
    """Blend with seamlessClone (NORMAL_CLONE); fall back to alpha blending."""
    dst = base_bgr.copy()
    dh, dw = dst.shape[:2]

    prepared = crop_to_clone_bounds(face_bgr, mask, center, dw, dh)
    if prepared is None:
        raise RuntimeError("Cannot place the aligned face without leaving the image.")
    face_bgr, mask, center = prepared

    if not overlay_fits(face_bgr, center, dw, dh):
        return alpha_blend(dst, face_bgr, mask, center)

    if mask.ndim == 3:
        mask_clone = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_clone = mask

    try:
        if cv2.countNonZero(mask_clone) == 0:
            raise RuntimeError("Empty face mask.")
        return cv2.seamlessClone(face_bgr, dst, mask_clone, center, cv2.NORMAL_CLONE)
    except cv2.error as exc:
        print(f"  seamlessClone failed ({exc}); falling back to alpha blending.")
        return alpha_blend(dst, face_bgr, mask_clone, center)
    except Exception as exc:
        print(f"  seamlessClone failed ({exc}); falling back to alpha blending.")
        return alpha_blend(dst, face_bgr, mask_clone, center)


def patch_from_aligned(
    warped_bgr: np.ndarray, dest_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    """Crop the warped face to the dest hull bbox for seamlessClone."""
    ys, xs = np.where(dest_mask > 16)
    if xs.size == 0 or ys.size == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad = max(2, int(0.04 * max(x1 - x0, y1 - y0)))
    h, w = warped_bgr.shape[:2]
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)

    face = warped_bgr[y0:y1, x0:x1].copy()
    mask = dest_mask[y0:y1, x0:x1].copy()
    if face.shape[0] < MIN_FACE_PX or face.shape[1] < MIN_FACE_PX:
        return None

    center = (x0 + face.shape[1] // 2, y0 + face.shape[0] // 2)
    return face, mask, center


# ---------------------------------------------------------------------------
# Batch loop
# ---------------------------------------------------------------------------
def process_sources(base_bgr: np.ndarray) -> None:
    """Align each source face onto the detected face in the template."""
    sources = list_source_images()
    if not sources:
        print(f"No images found in '{SOURCES_DIR}'. Add face photos and re-run.")
        return

    dh, dw = base_bgr.shape[:2]
    print(f"Processing {len(sources)} source image(s)...")

    with create_face_landmarker() as landmarker:
        dest_pts = detect_largest_landmarks(base_bgr, landmarker)
        if dest_pts is None:
            print(
                "ERROR: No face detected on base_image.jpg. "
                "Use a template that contains a clearly visible face.",
                file=sys.stderr,
            )
            sys.exit(1)

        dest_mask = hull_mask_from_landmarks(dest_pts, (dh, dw), dilate=True)
        dx, dy, dbw, dbh = cv2.boundingRect(dest_pts)
        print(f"Template face at ({dx}, {dy}) size {dbw}x{dbh}.")

        for src_path in sources:
            source = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if source is None:
                print(f"Skipping {src_path.name}: Could not read image")
                continue

            src_pts = detect_largest_landmarks(source, landmarker)
            if src_pts is None:
                print(f"Skipping {src_path.name}: No face detected")
                continue

            warped = align_source_to_base(source, src_pts, dest_pts, (dh, dw))
            warped = match_face_lighting(warped, base_bgr, dest_mask)
            patched = patch_from_aligned(warped, dest_mask)
            if patched is None:
                print(f"Skipping {src_path.name}: Aligned face region was empty")
                continue

            face_patch, mask_patch, center = patched
            try:
                merged = composite_face(base_bgr, face_patch, mask_patch, center)
            except Exception as exc:
                print(f"Skipping {src_path.name}: Composite failed ({exc})")
                continue

            out_name = f"merged_{src_path.stem}{src_path.suffix}"
            out_path = OUTPUTS_DIR / out_name
            if not cv2.imwrite(str(out_path), merged):
                print(f"Skipping {src_path.name}: Failed to write {out_path}")
                continue
            print(f"Wrote {out_path.name}")


def main() -> None:
    ensure_directories()
    base_bgr = load_base_image()
    process_sources(base_bgr)
    print("Done.")


if __name__ == "__main__":
    main()
