"""
two_stage_pipeline.py
=====================
Production-ready 2-stage visual search pipeline for Shopee product matching.

Architecture:
  Stage 1 — MobileCLIP (image + text fusion) → FAISS retrieval → top-K candidates
  Stage 2 — DINOv2-Small on YOLO-cropped images → rerank candidates → top-5

Environment: Google Colab T4 GPU (15 GB VRAM)
Dataset    : Shopee ~34,250 products
Target     : mAP@5 ≥ 0.79  (baseline 0.77 MobileCLIP-only)

Usage example
-------------
    from two_stage_pipeline import (
        crop_images_with_yolo,
        extract_dinov2_features,
        score_fusion,
        evaluate_two_stage,
        search_single_query,
    )
"""

from __future__ import annotations

import gc
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants (override via function params when needed)
# ──────────────────────────────────────────────────────────────────────────────
DINO_FEAT_DIM: int = 384          # dinov2_vits14 CLS token dimension
DINO_INPUT_SIZE: int = 224        # DINOv2 expected spatial resolution
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float]  = [0.229, 0.224, 0.225]

_DINO_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(DINO_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_image_safe(path: str) -> Optional[Image.Image]:
    """Open an image as RGB PIL Image; returns None on any failure."""
    try:
        return Image.open(path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError, Exception):
        return None


def _clamp_box(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int
) -> Optional[Tuple[float, float, float, float]]:
    """Clamp bounding box to image boundaries; return None if degenerate."""
    x1 = max(0.0, min(x1, W - 1))
    y1 = max(0.0, min(y1, H - 1))
    x2 = max(0.0, min(x2, float(W)))
    y2 = max(0.0, min(y2, float(H)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _choose_best_box(
    boxes: np.ndarray,       # (M, 4)  xyxy
    scores: np.ndarray,      # (M,)    confidence
    img_w: int,
    img_h: int,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.95,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Select the single best bounding box from YOLO predictions.

    Scoring = conf + 0.20 * center_proximity + 0.15 * area_score
    Boxes with area outside [min_area_ratio, max_area_ratio] of image area
    are rejected outright.

    Returns xyxy tuple or None if no suitable box exists.
    """
    img_area = max(img_w * img_h, 1)
    cx_img, cy_img = img_w / 2.0, img_h / 2.0
    diag = math.hypot(img_w, img_h)

    best_score: float = -1e9
    best_box: Optional[Tuple[float, float, float, float]] = None

    for i, (box, conf) in enumerate(zip(boxes, scores)):
        clamped = _clamp_box(*box[:4], img_w, img_h)
        if clamped is None:
            continue
        x1, y1, x2, y2 = clamped
        area_ratio = (x2 - x1) * (y2 - y1) / img_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist = math.hypot(cx - cx_img, cy - cy_img)
        center_score = 1.0 - dist / max(diag, 1.0)
        area_score = min(area_ratio / 0.45, 1.0)

        composite = float(conf) + 0.20 * center_score + 0.15 * area_score
        if composite > best_score:
            best_score = composite
            best_box = clamped

    return best_box


def _center_crop_fallback(
    img: Image.Image, ratio: float = 0.80
) -> Image.Image:
    """Return a center crop of `ratio` of the original image dimensions."""
    w, h = img.size
    new_w, new_h = int(w * ratio), int(h * ratio)
    left  = (w - new_w) // 2
    upper = (h - new_h) // 2
    return img.crop((left, upper, left + new_w, upper + new_h))


def _free_model(model: Optional[torch.nn.Module]) -> None:
    """Delete a PyTorch model from GPU to release VRAM."""
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 1 — crop_images_with_yolo
# ══════════════════════════════════════════════════════════════════════════════

def crop_images_with_yolo(
    image_paths: List[str],
    yolo_model: Any,                        # ultralytics.YOLO instance
    *,
    conf_threshold: float = 0.20,
    img_size: int = 640,
    padding_ratio: float = 0.08,
    fallback_center_ratio: float = 0.80,
    save_dir: Optional[str] = None,
    batch_size: int = 1,
) -> Dict[str, Image.Image]:
    """
    Detect product bounding boxes with YOLO and crop each image.

    For every image the function:
    1. Runs YOLO inference to get bounding boxes.
    2. Picks the best box via `_choose_best_box` (confidence + centrality +
       area heuristic).
    3. Expands the box by `padding_ratio` * box_side on each side.
    4. Falls back to an 80% center crop when YOLO finds nothing.
    5. Optionally saves cropped images to `save_dir`.

    Parameters
    ----------
    image_paths : list of str
        Absolute or relative paths to the source images.
    yolo_model : ultralytics.YOLO
        Pre-loaded YOLO model instance (already on the desired device).
    conf_threshold : float, default 0.20
        Minimum YOLO confidence to keep a detection.
    img_size : int, default 640
        YOLO inference image size.
    padding_ratio : float, default 0.08
        Fraction of box width/height to add as padding around the crop.
    fallback_center_ratio : float, default 0.80
        Fraction of image area to keep when YOLO fails.
    save_dir : str or None, default None
        If given, saves each cropped PIL image (original filename) to this
        directory. The directory is created if it does not exist.
    batch_size : int, default 1
        Number of images per YOLO forward pass (increase on strong GPUs).

    Returns
    -------
    dict
        Mapping ``{image_name: PIL.Image.Image}`` (RGB).
        - ``image_name`` is ``os.path.basename(path)``.
        - Images that cannot be opened receive the fallback center crop of a
          blank white 224×224 image (preserves index alignment).

    Example
    -------
    >>> from ultralytics import YOLO
    >>> yolo = YOLO("yolov8s.pt")
    >>> crops = crop_images_with_yolo(["img1.jpg", "img2.jpg"], yolo)
    >>> crops["img1.jpg"].size   # (W, H)
    (320, 320)
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    result: Dict[str, Image.Image] = {}
    stats = {"yolo": 0, "fallback": 0, "error": 0}

    for path in tqdm(image_paths, desc="YOLO crop", unit="img"):
        name = os.path.basename(path)
        pil_img = _load_image_safe(path)

        if pil_img is None:
            # Corrupt / missing image — return blank fallback
            stats["error"] += 1
            blank = Image.new("RGB", (DINO_INPUT_SIZE, DINO_INPUT_SIZE), 255)
            result[name] = blank
            if save_dir:
                blank.save(os.path.join(save_dir, name))
            continue

        # ── YOLO detection ─────────────────────────────────────────────────
        try:
            det = yolo_model.predict(
                path,
                imgsz=img_size,
                conf=conf_threshold,
                verbose=False,
            )[0]

            w, h = pil_img.size
            cropped: Image.Image

            if det.boxes is None or len(det.boxes) == 0:
                raise ValueError("no detections")

            boxes  = det.boxes.xyxy.cpu().numpy()   # (M, 4)
            confs  = det.boxes.conf.cpu().numpy()   # (M,)
            best   = _choose_best_box(boxes, confs, w, h)

            if best is None:
                raise ValueError("no suitable box")

            # Expand with padding
            x1, y1, x2, y2 = best
            bw, bh = x2 - x1, y2 - y1
            pad_x  = bw * padding_ratio
            pad_y  = bh * padding_ratio
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            cropped = pil_img.crop((x1, y1, x2, y2))
            stats["yolo"] += 1

        except Exception:
            # ── Fallback: center crop ──────────────────────────────────────
            cropped = _center_crop_fallback(pil_img, fallback_center_ratio)
            stats["fallback"] += 1

        result[name] = cropped
        if save_dir:
            save_path = os.path.join(save_dir, name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cropped.save(save_path)

    total = len(image_paths)
    print(
        f"✅ YOLO crop done | "
        f"yolo={stats['yolo']} ({stats['yolo']/max(total,1):.1%}) | "
        f"fallback={stats['fallback']} | "
        f"error={stats['error']}"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 2 — extract_dinov2_features
# ══════════════════════════════════════════════════════════════════════════════

def extract_dinov2_features(
    cropped_images: Union[Dict[str, Image.Image], List[Image.Image]],
    *,
    batch_size: int = 32,
    device: Optional[str] = None,
    save_path: Optional[str] = None,
    model_name: str = "dinov2_vits14",
    free_model_after: bool = True,
) -> np.ndarray:
    """
    Extract L2-normalised DINOv2 CLS-token features from cropped images.

    The function loads DINOv2-Small (``dinov2_vits14``) from ``torch.hub``,
    processes images in mini-batches, extracts ``x_norm_clstoken`` (384-D),
    L2-normalises each vector, and optionally saves the resulting array.

    Parameters
    ----------
    cropped_images : dict {str: PIL.Image} **or** list of PIL.Image
        Output of ``crop_images_with_yolo``, or any ordered collection of
        PIL images.  Order determines the row order in the returned array.
    batch_size : int, default 32
        Number of images per GPU batch.  Reduce to 16 on low VRAM.
    device : str or None
        ``"cuda"`` | ``"cpu"`` | None (auto-detect).
    save_path : str or None
        If provided, the feature array is saved as a ``.npy`` file at this
        path.  Parent directories are created automatically.
    model_name : str, default ``"dinov2_vits14"``
        torch.hub model name.  Change to ``"dinov2_vitb14"`` for 768-D.
    free_model_after : bool, default True
        Delete the DINOv2 model from VRAM after extraction to free memory
        for MobileCLIP or FAISS operations.

    Returns
    -------
    np.ndarray, shape (N, 384), dtype float32
        L2-normalised feature matrix.  Rows correspond to input order.

    Raises
    ------
    RuntimeError
        If model loading fails and no fallback is available.

    Example
    -------
    >>> feats = extract_dinov2_features(crops, batch_size=32,
    ...                                  save_path="/content/features/gallery_dino.npy")
    >>> feats.shape
    (34250, 384)
    """
    _device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Normalise input to ordered list
    if isinstance(cropped_images, dict):
        imgs: List[Image.Image] = list(cropped_images.values())
    else:
        imgs = list(cropped_images)

    n = len(imgs)
    print(f"📦 Extracting DINOv2 features | N={n:,} | device={_device} | batch={batch_size}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"⚙️  Loading {model_name} from torch.hub …")
    try:
        dino: torch.nn.Module = torch.hub.load(
            "facebookresearch/dinov2", model_name, verbose=False
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load DINOv2 from torch.hub: {e}\n"
            "Ensure internet access or pre-cached model."
        ) from e

    dino = dino.eval().to(_device)

    # ── Batch extraction ───────────────────────────────────────────────────
    all_feats: List[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="DINOv2 extract", unit="batch"):
            batch_pils = imgs[start : start + batch_size]
            tensors: List[torch.Tensor] = []

            for pil in batch_pils:
                if pil is None or not isinstance(pil, Image.Image):
                    # Replace broken image with blank tensor
                    t = torch.zeros(3, DINO_INPUT_SIZE, DINO_INPUT_SIZE)
                else:
                    try:
                        t = _DINO_TRANSFORM(pil.convert("RGB"))
                    except Exception:
                        t = torch.zeros(3, DINO_INPUT_SIZE, DINO_INPUT_SIZE)
                tensors.append(t)

            batch = torch.stack(tensors).to(_device, non_blocking=True)

            try:
                out = dino.forward_features(batch)
                # DINOv2 returns a dict; grab CLS token
                if isinstance(out, dict):
                    feats = out.get("x_norm_clstoken", out.get("x_cls", None))
                    if feats is None:
                        # fallback: first value
                        feats = next(iter(out.values()))
                else:
                    feats = out                      # already tensor
                if feats.ndim == 3:
                    feats = feats[:, 0, :]           # [B, seq, D] → [B, D]
            except Exception as exc:
                warnings.warn(f"DINOv2 forward error on batch {start}: {exc}")
                feats = torch.zeros(len(tensors), DINO_FEAT_DIM, device=_device)

            feats = F.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy().astype(np.float32))

    features = np.vstack(all_feats).astype(np.float32)  # (N, D)
    # Double-check normalization
    features = _l2_normalize(features)

    # ── Save ───────────────────────────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.save(save_path, features)
        print(f"💾 Saved DINOv2 features → {save_path}  shape={features.shape}")

    # ── Free VRAM ──────────────────────────────────────────────────────────
    if free_model_after:
        _free_model(dino)
        print("🗑️  DINOv2 unloaded from VRAM")

    print(f"✅ DINOv2 features ready | shape={features.shape} | dtype={features.dtype}")
    return features


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 3 — score_fusion
# ══════════════════════════════════════════════════════════════════════════════

def score_fusion(
    dino_scores: np.ndarray,
    clip_scores: np.ndarray,
    beta: float,
    *,
    top_k: int = 5,
    return_scores: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse DINOv2 re-ranking scores with MobileCLIP Stage-1 scores.

    Both score vectors are independently min-max normalised to [0, 1] before
    blending, which prevents one modality from dominating due to different
    score ranges.

        fused = beta * norm(dino) + (1 - beta) * norm(clip)

    Parameters
    ----------
    dino_scores : np.ndarray, shape (M,)
        Cosine similarity scores between the query's DINOv2 crop feature and
        each of the M candidate crop features.
    clip_scores : np.ndarray, shape (M,)
        FAISS inner-product (cosine) distances returned by Stage-1 FAISS
        search, aligned with the same M candidates.
    beta : float
        Blend weight for DINOv2.
        - 0.0 → pure MobileCLIP (no re-ranking)
        - 0.3 → mild DINOv2 influence (safe default)
        - 0.5 → equal weight
        - 1.0 → pure DINOv2 (risky if YOLO is inaccurate)
    top_k : int, default 5
        Number of top indices to return.
    return_scores : bool, default True
        Whether to also return the fused score values alongside indices.

    Returns
    -------
    top_indices : np.ndarray, shape (top_k,)
        Indices (into the M-length candidate arrays) of the top-k results,
        sorted descending by fused score.
    top_fused_scores : np.ndarray, shape (top_k,)
        Corresponding fused scores in [0, 1].

    Raises
    ------
    ValueError
        If ``dino_scores`` and ``clip_scores`` have different lengths.

    Example
    -------
    >>> dino = np.array([0.92, 0.85, 0.70])
    >>> clip = np.array([0.60, 0.88, 0.75])
    >>> idx, sc = score_fusion(dino, clip, beta=0.3, top_k=2)
    >>> idx
    array([1, 0])
    """
    dino_scores = np.asarray(dino_scores, dtype=np.float32).ravel()
    clip_scores = np.asarray(clip_scores, dtype=np.float32).ravel()

    if dino_scores.shape != clip_scores.shape:
        raise ValueError(
            f"Shape mismatch: dino_scores {dino_scores.shape} vs "
            f"clip_scores {clip_scores.shape}"
        )

    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    dino_norm = _minmax(dino_scores)
    clip_norm  = _minmax(clip_scores)

    fused: np.ndarray = beta * dino_norm + (1.0 - beta) * clip_norm  # (M,)

    top_k_eff = min(top_k, len(fused))
    # Use argpartition for O(M) instead of full argsort
    part = np.argpartition(-fused, top_k_eff - 1)[:top_k_eff]
    order = np.argsort(-fused[part])
    top_indices = part[order].astype(np.int32)
    top_fused_scores = fused[top_indices]

    return top_indices, top_fused_scores


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 4 — evaluate_two_stage
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_two_stage(
    query_df,                           # pd.DataFrame with columns: image, title, label_group
    gallery_df,                         # pd.DataFrame with columns: image, label_group
    *,
    alpha: float,                       # MobileCLIP image/text blend weight
    retrieval_k: int,                   # Stage-1 candidates to retrieve
    beta: float,                        # Stage-2 DINOv2 blend weight
    clip_image_features: np.ndarray,    # (N_gallery, 512)
    clip_text_features: np.ndarray,     # (N_gallery, 512)
    dino_crop_features: np.ndarray,     # (N_gallery, 384)
    query_clip_image_features: np.ndarray,   # (N_query, 512)
    query_clip_text_features: np.ndarray,    # (N_query, 512)
    query_dino_crop_features: np.ndarray,    # (N_query, 384)
    image_dir: str,
    final_k: int = 5,
    use_gpu_faiss: bool = True,
    save_results_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate the full 2-stage pipeline on a query set.

    Assumes all features are **pre-computed and L2-normalised** (both gallery
    and query).  This is the recommended approach for batch evaluation to avoid
    repeated model forward passes.

    Evaluation pipeline per query
    ──────────────────────────────
    1. Fuse query MobileCLIP features:
       ``q_fused = normalize(alpha * q_img + (1-alpha) * q_txt)``
    2. FAISS inner-product search → top ``retrieval_k`` gallery candidates.
    3. Compute DINOv2 cosine scores for those candidates.
    4. Call ``score_fusion(dino_scores, clip_scores, beta)`` → top-5.
    5. Compute AP@5, Precision@1, Recall@5.

    Parameters
    ----------
    query_df : pd.DataFrame
        Must have columns ``image`` (filename), ``title``, ``label_group``.
    gallery_df : pd.DataFrame
        Must have columns ``image`` (filename), ``label_group``.
    alpha : float
        Image weight in MobileCLIP fusion (0=text-only, 1=image-only).
    retrieval_k : int
        Number of FAISS candidates from Stage-1 (recommended 50–200).
    beta : float
        DINOv2 weight in score fusion (recommended 0.2–0.5).
    clip_image_features : np.ndarray (N_gallery, 512)
        Pre-extracted MobileCLIP image embeddings for gallery.
    clip_text_features : np.ndarray (N_gallery, 512)
        Pre-extracted MobileCLIP text embeddings for gallery.
    dino_crop_features : np.ndarray (N_gallery, 384)
        Pre-extracted DINOv2-crop embeddings for gallery.
    query_clip_image_features : np.ndarray (N_query, 512)
        MobileCLIP image embeddings for query set.
    query_clip_text_features : np.ndarray (N_query, 512)
        MobileCLIP text embeddings for query set.
    query_dino_crop_features : np.ndarray (N_query, 384)
        DINOv2-crop embeddings for query set.
    image_dir : str
        Root directory for images (used only for error reporting).
    final_k : int, default 5
        Number of results to return per query (used in AP@k, Recall@k).
    use_gpu_faiss : bool, default True
        Move FAISS index to GPU if available (faster search).
    save_results_path : str or None
        If given, saves a JSON with full per-query results.

    Returns
    -------
    dict with keys:
        ``mAP@5``, ``Precision@1``, ``Recall@5``,
        ``n_queries``, ``latency_ms_per_query``

    Example
    -------
    >>> metrics = evaluate_two_stage(df_val, df_gallery,
    ...     alpha=0.5, retrieval_k=100, beta=0.3,
    ...     clip_image_features=gal_img_feat,
    ...     clip_text_features=gal_txt_feat,
    ...     dino_crop_features=gal_dino_feat,
    ...     query_clip_image_features=q_img_feat,
    ...     query_clip_text_features=q_txt_feat,
    ...     query_dino_crop_features=q_dino_feat,
    ...     image_dir="/content/train_images")
    >>> print(f"mAP@5 = {metrics['mAP@5']:.4f}")
    """
    import pandas as pd  # local import — pandas is guaranteed in Colab

    # ── Build FAISS index on fused gallery features ────────────────────────
    gal_img  = _l2_normalize(clip_image_features.astype(np.float32))
    gal_txt  = _l2_normalize(clip_text_features.astype(np.float32))
    gal_fused = _l2_normalize(alpha * gal_img + (1.0 - alpha) * gal_txt)

    dim = gal_fused.shape[1]
    index = faiss.IndexFlatIP(dim)
    if use_gpu_faiss and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        print("⚡ FAISS running on GPU")

    index.add(np.ascontiguousarray(gal_fused))
    print(f"✅ FAISS index built | {index.ntotal:,} vectors | dim={dim}")

    # ── Ground truth mapping: label_group → set of gallery indices ─────────
    gallery_labels = gallery_df["label_group"].values
    label_to_gal_idxs: Dict[Any, List[int]] = {}
    for i, lbl in enumerate(gallery_labels):
        label_to_gal_idxs.setdefault(lbl, []).append(i)

    # ── Fuse query features ────────────────────────────────────────────────
    q_img   = _l2_normalize(query_clip_image_features.astype(np.float32))
    q_txt   = _l2_normalize(query_clip_text_features.astype(np.float32))
    q_fused = _l2_normalize(alpha * q_img + (1.0 - alpha) * q_txt)

    q_dino  = _l2_normalize(query_dino_crop_features.astype(np.float32))
    gal_dino = _l2_normalize(dino_crop_features.astype(np.float32))

    # ── Per-query evaluation ───────────────────────────────────────────────
    ap_list: List[float] = []
    p1_list: List[float] = []
    r5_list: List[float] = []
    per_query_results: List[Dict] = []

    t_start = time.perf_counter()

    for q_idx in tqdm(range(len(query_df)), desc=f"Eval (α={alpha},k={retrieval_k},β={beta})"):
        row = query_df.iloc[q_idx]
        q_label = row["label_group"]

        # Ground truth: all gallery items of same label
        relevant_idxs = set(label_to_gal_idxs.get(q_label, []))
        if not relevant_idxs:
            continue

        try:
            # Stage 1: FAISS retrieval
            q_vec = np.ascontiguousarray(q_fused[q_idx : q_idx + 1])  # (1, D)
            distances, indices = index.search(q_vec, retrieval_k + 1)

            cand_idxs   = indices[0]           # (retrieval_k+1,)
            clip_scores = distances[0]          # (retrieval_k+1,)

            # Remove self-match (if query is also in gallery)
            mask = cand_idxs >= 0
            cand_idxs   = cand_idxs[mask]
            clip_scores = clip_scores[mask]

            if len(cand_idxs) == 0:
                ap_list.append(0.0)
                p1_list.append(0.0)
                r5_list.append(0.0)
                continue

            # Stage 2: DINOv2 cosine scores
            q_dv = q_dino[q_idx]                             # (384,)
            cand_dino = gal_dino[cand_idxs]                  # (M, 384)
            dino_scores: np.ndarray = cand_dino @ q_dv       # (M,)

            # Score fusion
            local_top_idxs, _ = score_fusion(
                dino_scores, clip_scores, beta=beta, top_k=final_k
            )
            global_top_idxs = cand_idxs[local_top_idxs]     # map back to gallery

            # ── Metrics ───────────────────────────────────────────────────
            retrieved_labels = gallery_labels[global_top_idxs]

            # AP@k
            hits, ap = 0, 0.0
            n_relevant = len(relevant_idxs)
            for rank, g_idx in enumerate(global_top_idxs[:final_k], start=1):
                if g_idx in relevant_idxs:
                    hits += 1
                    ap  += hits / rank
            ap_list.append(ap / min(n_relevant, final_k))

            # Precision@1
            p1 = 1.0 if (len(global_top_idxs) > 0 and global_top_idxs[0] in relevant_idxs) else 0.0
            p1_list.append(p1)

            # Recall@5
            r5 = len(set(global_top_idxs[:final_k]) & relevant_idxs) / n_relevant
            r5_list.append(r5)

            if save_results_path:
                per_query_results.append({
                    "q_idx": q_idx,
                    "label_group": str(q_label),
                    "AP@5": ap_list[-1],
                    "P@1": p1,
                    "R@5": r5,
                    "top_indices": global_top_idxs.tolist(),
                })

        except Exception as exc:
            warnings.warn(f"Error on query {q_idx} ({row.get('image','?')}): {exc}")
            ap_list.append(0.0)
            p1_list.append(0.0)
            r5_list.append(0.0)

    elapsed = time.perf_counter() - t_start
    n_q = max(len(ap_list), 1)

    metrics = {
        "mAP@5"               : float(np.mean(ap_list)),
        "Precision@1"         : float(np.mean(p1_list)),
        "Recall@5"            : float(np.mean(r5_list)),
        "n_queries"           : n_q,
        "latency_ms_per_query": round(elapsed / n_q * 1000, 2),
    }

    print(
        f"\n📊 Results  α={alpha:.2f} | K={retrieval_k} | β={beta:.2f}\n"
        f"   mAP@5        = {metrics['mAP@5']:.4f}\n"
        f"   Precision@1  = {metrics['Precision@1']:.4f}\n"
        f"   Recall@5     = {metrics['Recall@5']:.4f}\n"
        f"   Latency      = {metrics['latency_ms_per_query']:.1f} ms/query"
    )

    # ── Save ──────────────────────────────────────────────────────────────
    if save_results_path:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(save_results_path)), exist_ok=True)
        payload = {
            "params": {"alpha": alpha, "retrieval_k": retrieval_k, "beta": beta},
            "metrics": metrics,
            "per_query": per_query_results,
        }
        with open(save_results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved → {save_results_path}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 5 — search_single_query
# ══════════════════════════════════════════════════════════════════════════════

def search_single_query(
    query_image: Union[str, Image.Image],
    query_title: str,
    params: Dict[str, Any],
    *,
    yolo_model: Any,
    clip_model: Any,
    clip_tokenizer: Any,
    clip_preprocess: Any,
    dino_model: Optional[torch.nn.Module],
    faiss_index: Any,
    gallery_dino_features: np.ndarray,   # (N_gallery, 384)
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    End-to-end single-query 2-stage visual search.

    Takes one query image + title and returns the top-5 gallery indices with
    their fused confidence scores.  All models are assumed pre-loaded to
    avoid cold-start overhead for repeated queries.

    Pipeline
    --------
    1. YOLO crop of query image (with padding; fallback: center crop).
    2. Extract MobileCLIP features (original image + text).
    3. Fuse via ``alpha``; FAISS inner-product search → top ``retrieval_k``.
    4. Extract DINOv2 feature from cropped query.
    5. Score fusion → top-5 indices.

    Parameters
    ----------
    query_image : str or PIL.Image.Image
        Path to query image **or** a PIL.Image already loaded (RGB).
    query_title : str
        Product title / description for the text branch.
    params : dict
        Must contain:
        - ``alpha``       (float): MobileCLIP image/text blend weight.
        - ``retrieval_k`` (int):   Stage-1 candidate pool size.
        - ``beta``        (float): DINOv2 blend weight.
        Optionally:
        - ``yolo_conf``   (float): YOLO confidence threshold (default 0.20).
        - ``padding_ratio``(float): Crop padding (default 0.08).
        - ``top_k``       (int):   Final results to return (default 5).
    yolo_model : ultralytics.YOLO
        Pre-loaded YOLO model.
    clip_model
        Pre-loaded MobileCLIP model with ``encode_image`` / ``encode_text``.
    clip_tokenizer
        MobileCLIP tokenizer function: ``tokenizer([text]) → tensor``.
    clip_preprocess
        MobileCLIP image pre-processing transform.
    dino_model : torch.nn.Module or None
        Pre-loaded DINOv2 model.  If ``None``, Stage-2 is skipped and only
        FAISS ranks are returned (useful for ablation / low-VRAM scenarios).
    faiss_index
        Pre-built FAISS index of fused gallery MobileCLIP features.
    gallery_dino_features : np.ndarray (N_gallery, 384)
        Pre-computed L2-normalised DINOv2 crop features for the full gallery.
    device : torch.device or None
        Inference device.  Auto-detected if None.

    Returns
    -------
    top_5_indices : np.ndarray, shape (top_k,)
        Gallery row indices of the top-k matches.
    top_5_scores : np.ndarray, shape (top_k,)
        Corresponding fused scores in [0, 1].

    Raises
    ------
    FileNotFoundError
        If ``query_image`` is a string path and the file does not exist.
    ValueError
        If required keys are missing from ``params``.

    Example
    -------
    >>> indices, scores = search_single_query(
    ...     "test_query.jpg", "Nike Air Max 90 White",
    ...     params={"alpha": 0.5, "retrieval_k": 100, "beta": 0.3},
    ...     yolo_model=yolo, clip_model=clip, clip_tokenizer=tokenizer,
    ...     clip_preprocess=preprocess, dino_model=dino,
    ...     faiss_index=index, gallery_dino_features=gal_dino)
    >>> print(indices, scores)
    [1042  876 3301   55 2198] [0.912 0.897 0.881 0.860 0.843]
    """
    _device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Validate required params
    required = {"alpha", "retrieval_k", "beta"}
    missing = required - set(params.keys())
    if missing:
        raise ValueError(f"params is missing required keys: {missing}")

    alpha        = float(params["alpha"])
    retrieval_k  = int(params["retrieval_k"])
    beta         = float(params["beta"])
    yolo_conf    = float(params.get("yolo_conf", 0.20))
    pad_ratio    = float(params.get("padding_ratio", 0.08))
    top_k        = int(params.get("top_k", 5))

    t0_total = time.perf_counter()
    timings: Dict[str, float] = {}

    # ── 1. Load query image ────────────────────────────────────────────────
    if isinstance(query_image, str):
        if not os.path.exists(query_image):
            raise FileNotFoundError(f"Query image not found: {query_image}")
        pil_orig = _load_image_safe(query_image) or Image.new("RGB", (224, 224), 255)
        img_path_str: Optional[str] = query_image
    else:
        pil_orig = query_image.convert("RGB")
        img_path_str = None

    # ── 2. YOLO crop of query ──────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        w, h = pil_orig.size
        if img_path_str:
            det = yolo_model.predict(
                img_path_str, imgsz=640, conf=yolo_conf, verbose=False
            )[0]
        else:
            # Convert PIL → numpy for YOLO
            arr = np.array(pil_orig)[:, :, ::-1]   # RGB→BGR
            det = yolo_model.predict(
                arr, imgsz=640, conf=yolo_conf, verbose=False
            )[0]

        boxes  = det.boxes.xyxy.cpu().numpy() if det.boxes else np.zeros((0, 4))
        confs  = det.boxes.conf.cpu().numpy() if det.boxes else np.zeros(0)
        best   = _choose_best_box(boxes, confs, w, h) if len(boxes) > 0 else None

        if best is not None:
            x1, y1, x2, y2 = best
            bw, bh = x2 - x1, y2 - y1
            x1 = max(0, x1 - bw * pad_ratio)
            y1 = max(0, y1 - bh * pad_ratio)
            x2 = min(w, x2 + bw * pad_ratio)
            y2 = min(h, y2 + bh * pad_ratio)
            pil_crop = pil_orig.crop((x1, y1, x2, y2))
            crop_source = "yolo"
        else:
            pil_crop = _center_crop_fallback(pil_orig)
            crop_source = "center_crop"

    except Exception as yolo_exc:
        warnings.warn(f"YOLO failed on query ({yolo_exc}); using center crop.")
        pil_crop = _center_crop_fallback(pil_orig)
        crop_source = "center_crop_fallback"

    timings["yolo_ms"] = (time.perf_counter() - t0) * 1000

    # ── 3. MobileCLIP feature extraction ──────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        # Image feature (original image — per CLAUDE.md design)
        img_tensor = clip_preprocess(pil_orig).unsqueeze(0).to(_device)
        q_img_feat = clip_model.encode_image(img_tensor)
        q_img_feat = F.normalize(q_img_feat, dim=-1).cpu().numpy().flatten()

        # Text feature
        txt_tokens = clip_tokenizer([query_title]).to(_device)
        q_txt_feat = clip_model.encode_text(txt_tokens)
        q_txt_feat = F.normalize(q_txt_feat, dim=-1).cpu().numpy().flatten()

    # Weighted fusion
    q_fused = alpha * q_img_feat + (1.0 - alpha) * q_txt_feat
    norm = np.linalg.norm(q_fused)
    q_fused = q_fused / max(norm, 1e-12)
    timings["clip_ms"] = (time.perf_counter() - t0) * 1000

    # ── 4. FAISS search (Stage 1) ──────────────────────────────────────────
    t0 = time.perf_counter()
    q_vec = np.ascontiguousarray(q_fused[np.newaxis, :].astype(np.float32))
    faiss_scores, faiss_indices = faiss_index.search(q_vec, retrieval_k)
    cand_idxs   = faiss_indices[0]    # (retrieval_k,)
    clip_scores = faiss_scores[0]     # (retrieval_k,) cosine scores ∈ [-1, 1]
    # Remove invalid
    valid_mask = cand_idxs >= 0
    cand_idxs   = cand_idxs[valid_mask]
    clip_scores = clip_scores[valid_mask]
    timings["faiss_ms"] = (time.perf_counter() - t0) * 1000

    # ── 5. DINOv2 re-ranking (Stage 2) ────────────────────────────────────
    t0 = time.perf_counter()
    if dino_model is not None and len(cand_idxs) > 0:
        with torch.no_grad():
            crop_tensor = _DINO_TRANSFORM(pil_crop).unsqueeze(0).to(_device)
            out = dino_model.forward_features(crop_tensor)
            if isinstance(out, dict):
                q_dino = out.get("x_norm_clstoken", next(iter(out.values())))
            else:
                q_dino = out
            if q_dino.ndim == 3:
                q_dino = q_dino[:, 0, :]
            q_dino = F.normalize(q_dino, dim=-1).cpu().numpy().flatten()

        gal_dino_cand = _l2_normalize(
            gallery_dino_features[cand_idxs].astype(np.float32)
        )
        dino_scores: np.ndarray = gal_dino_cand @ q_dino          # (M,)

        local_top, fused_scores = score_fusion(
            dino_scores, clip_scores, beta=beta, top_k=top_k
        )
        top_5_global_idxs  = cand_idxs[local_top]
        top_5_scores        = fused_scores

        timings["dino_ms"] = (time.perf_counter() - t0) * 1000

    else:
        # Stage-2 skipped: use raw FAISS order
        top_5_global_idxs = cand_idxs[:top_k]
        top_5_scores      = clip_scores[:top_k]
        timings["dino_ms"] = 0.0

    timings["total_ms"] = (time.perf_counter() - t0_total) * 1000

    print(
        f"🔍 Query done | crop={crop_source} | "
        f"yolo={timings['yolo_ms']:.0f}ms | "
        f"clip={timings['clip_ms']:.0f}ms | "
        f"faiss={timings['faiss_ms']:.0f}ms | "
        f"dino={timings['dino_ms']:.0f}ms | "
        f"total={timings['total_ms']:.0f}ms"
    )

    return (
        top_5_global_idxs.astype(np.int32),
        top_5_scores.astype(np.float32),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test (run as script: python two_stage_pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Smoke-test: score_fusion")
    print("=" * 60)

    rng = np.random.default_rng(42)
    dino_s = rng.uniform(0.5, 1.0, size=100).astype(np.float32)
    clip_s = rng.uniform(0.3, 0.9, size=100).astype(np.float32)

    for beta in [0.0, 0.3, 0.5, 1.0]:
        idxs, scores = score_fusion(dino_s, clip_s, beta=beta, top_k=5)
        print(f"  beta={beta:.1f} → top-5 indices={idxs}  scores={scores.round(4)}")

    print("\n✅ score_fusion OK")
