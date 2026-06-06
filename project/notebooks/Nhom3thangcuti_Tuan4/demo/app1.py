# ============================================================
# app.py – Shopee Visual Search Demo
# Khớp với pipeline chính: MobileCLIP/CLIP + FAISS + DINOv2 re-ranking
# Không dùng top_indices.npy. Dùng đúng cache pipeline sinh ra:
#   - train.csv
#   - mobileclip_gallery_img.npy
#   - mobileclip_gallery_txt.npy
#   - dinov2_gallery.npy
#   - final_metric_dinov2.csv (nếu có, để lấy alpha và retrieval_k)
# ============================================================

import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    import faiss
except Exception as exc:
    faiss = None

# ============================================================
# 1. CẤU HÌNH
# ============================================================

st.set_page_config(
    page_title="Shopee Visual Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Chỉnh ở đây nếu app không tự dò được đường dẫn.
# Để chuỗi rỗng "" nếu muốn tự dò.
# Nếu máy em giống notebook/app cũ thì để sẵn path này. Sai thì sửa 4 dòng này.
MANUAL_DATA_DIR = r"E:\study\DuLieuPython"        # thư mục chứa train.csv
MANUAL_IMAGE_DIR = r"E:\study\DuLieuPython\shopee-product-matching\train_images"       # thư mục chứa ảnh train_images
MANUAL_FEATURE_DIR = r""     # thư mục chứa mobileclip_gallery_img.npy, mobileclip_gallery_txt.npy, dinov2_gallery.npy
MANUAL_METRIC_CSV = r""      # final_metric_dinov2.csv nếu có

# Tham số mặc định nếu không đọc được từ final_metric_dinov2.csv
DEFAULT_ALPHA = 0.5
DEFAULT_RETRIEVAL_K = 100
DEFAULT_BETA = 0.30
FINAL_K = 5

# MobileCLIP giống pipeline chính
MOBILECLIP_VARIANT = "mobileclip_s0"
MOBILECLIP_CKPT = "/tmp/mobileclip_s0.pt"
HF_MODEL = "openai/clip-vit-base-patch32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 2. CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }
    .hero-banner {
        background: linear-gradient(135deg, #EE4D2D 0%, #FF7337 100%);
        border-radius: 16px; padding: 30px 36px; margin-bottom: 26px;
        display: flex; align-items: center; gap: 18px;
        box-shadow: 0 8px 30px rgba(238,77,45,0.30);
    }
    .hero-title { font-size: 2rem; font-weight: 700; color: #fff; line-height: 1.2; margin: 0; }
    .hero-sub { font-size: 0.95rem; color: rgba(255,255,255,0.90); margin: 6px 0 0; }
    .hero-icon { font-size: 3rem; }
    .section-label { font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: #EE4D2D; margin-bottom: 10px; }
    .query-card { background: #fff8f6; border: 2px solid #ffd5c8; border-radius: 14px; padding: 18px; text-align: center; }
    .result-card { background: #fff; border: 1.5px solid #f0e8e6; border-radius: 12px; padding: 12px; text-align: center; height: 100%; }
    .rank-badge { display: inline-block; background: #EE4D2D; color: #fff; font-size: .72rem; font-weight: 700; border-radius: 20px; padding: 2px 10px; margin-bottom: 8px; }
    .img-name { font-size: .73rem; color: #666; word-break: break-all; margin-top: 6px; }
    .score-badge { display: inline-block; background: #fff3f0; color: #EE4D2D; border: 1px solid #ffcbbf; font-size: .72rem; font-weight: 600; border-radius: 20px; padding: 2px 10px; margin-top: 4px; }
    .correct-tag { color: #27ae60; font-weight: 700; }
    .wrong-tag { color: #e74c3c; font-weight: 700; }
    div.stButton > button {
        background: linear-gradient(135deg, #EE4D2D, #FF7337) !important;
        color: #fff !important; font-weight: 700 !important;
        border: none !important; border-radius: 10px !important;
        padding: .6rem 2.2rem !important; width: 100%;
    }
    [data-testid="stFileUploader"] { border: 2px dashed #ffbfaf !important; border-radius: 12px !important; background: #fff8f6 !important; padding: 12px !important; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-banner">
        <span class="hero-icon">🔍</span>
        <div>
            <p class="hero-title">Shopee Visual Search</p>
            <p class="hero-sub">MobileCLIP/CLIP + FAISS candidate generation · DINOv2 re-ranking</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. HELPER DÒ FILE
# ============================================================

def existing_path(p: str) -> Optional[Path]:
    if not p or not str(p).strip():
        return None
    path = Path(p)
    return path if path.exists() else None


def unique_existing_dirs(dirs: List[Path]) -> List[Path]:
    out, seen = [], set()
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        key = str(d.resolve())
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def app_roots() -> List[Path]:
    root = Path(__file__).parent.resolve()
    roots = [root] + list(root.parents[:8])
    extra = [
        Path.cwd(),
        Path("/content"),
        Path("/content/features"),
        Path("/content/drive/MyDrive/DoAnPython/DuLieuPython"),
        Path("/content/drive/MyDrive/DuLieuPython"),
        Path("/content/drive/My Drive/DoAnPython/DuLieuPython"),
        Path("/content/drive/My Drive/DuLieuPython"),
    ]
    return unique_existing_dirs(roots + extra)


def find_latest(patterns: List[str], manual_path: str = "", required: bool = True, name: str = "file") -> Optional[Path]:
    manual = existing_path(manual_path)
    if manual is not None:
        return manual

    matches = []
    search_dirs = []
    for root in app_roots():
        search_dirs += [
            root,
            root / "features",
            root / "data",
            root / "data" / "processed",
            root / "data" / "raw",
            root / "outputs",
            root / "results",
            root / "notebooks",
        ]
    search_dirs = unique_existing_dirs(search_dirs)

    for base in search_dirs:
        for pat in patterns:
            try:
                matches.extend(p for p in base.glob(pat) if p.is_file())
            except Exception:
                pass

    uniq = {str(p.resolve()): p for p in matches}
    matches = sorted(uniq.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        if required:
            raise FileNotFoundError(f"Không tìm thấy {name}. Hãy chỉnh MANUAL_* ở đầu app.py.")
        return None
    return matches[0]


def find_data_dir() -> Path:
    manual = existing_path(MANUAL_DATA_DIR)
    if manual and (manual / "train.csv").exists():
        return manual

    candidates = []
    for root in app_roots():
        candidates += [
            root,
            root / "DuLieuPython",
            root / "DoAnPython" / "DuLieuPython",
            root / "data",
            root / "data" / "raw",
            root / "project" / "data" / "raw",
        ]
    for c in unique_existing_dirs(candidates):
        if (c / "train.csv").exists():
            return c
    raise FileNotFoundError("Không tìm thấy train.csv. Chỉnh MANUAL_DATA_DIR.")


def find_image_dir(data_dir: Path) -> Path:
    manual = existing_path(MANUAL_IMAGE_DIR)
    if manual and manual.is_dir():
        return manual

    candidates = [
        data_dir / "train_images",
        data_dir.parent / "train_images",
        data_dir / "data" / "raw" / "train_images",
        data_dir / "project" / "data" / "raw" / "train_images",
        Path(r"E:\study\DuLieuPython\data\raw\train_images"),
        Path(r"D:\study\DuLieuPython\data\raw\train_images"),
        Path(r"E:\Study\DuLieuPython\data\raw\train_images"),
        Path(r"D:\Study\DuLieuPython\data\raw\train_images"),
        Path("/content/train_images_extracted/train_images"),
        Path("/content/drive/MyDrive/DoAnPython/DuLieuPython/train_images"),
        Path("/content/drive/MyDrive/DuLieuPython/train_images"),
        Path("/content/drive/MyDrive/DuLieuPython/data/raw/train_images"),
    ]
    for root in app_roots():
        candidates += [
            root / "train_images",
            root / "data" / "raw" / "train_images",
            root / "project" / "data" / "raw" / "train_images",
        ]
    for c in unique_existing_dirs(candidates):
        if c.is_dir():
            return c
    raise FileNotFoundError(
        "Không tìm thấy thư mục train_images. Sửa MANUAL_IMAGE_DIR ở đầu app.py thành đường dẫn thật, "
        "ví dụ: r'E:\\study\\DuLieuPython\\data\\raw\\train_images'."
    )


def find_feature_dir() -> Path:
    manual = existing_path(MANUAL_FEATURE_DIR)
    if manual and manual.is_dir():
        return manual

    required = {"mobileclip_gallery_img.npy", "mobileclip_gallery_txt.npy", "dinov2_gallery.npy"}
    candidates = []
    for root in app_roots():
        candidates += [root, root / "features", root / "content" / "features", root / "outputs", root / "results"]

    for c in unique_existing_dirs(candidates):
        if all((c / f).exists() for f in required):
            return c
    raise FileNotFoundError(
        "Không tìm thấy đủ feature cache: mobileclip_gallery_img.npy, "
        "mobileclip_gallery_txt.npy, dinov2_gallery.npy. Hãy chạy pipeline chính trước."
    )


def resolve_image_path(image_name: str, image_dir: Path) -> str:
    name = str(image_name)
    candidates = [Path(name), image_dir / name, image_dir / Path(name).name]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(image_dir / Path(name).name)

# ============================================================
# 4. LOAD PARAMS TỪ final_metric_dinov2.csv
# ============================================================

def load_pipeline_params() -> Dict[str, Any]:
    alpha = DEFAULT_ALPHA
    retrieval_k = DEFAULT_RETRIEVAL_K
    beta = DEFAULT_BETA
    metric_csv = find_latest(
        patterns=["**/final_metric_dinov2.csv", "**/final_metric.csv"],
        manual_path=MANUAL_METRIC_CSV,
        required=False,
        name="final_metric_dinov2.csv",
    )
    if metric_csv is not None:
        try:
            dfm = pd.read_csv(metric_csv)
            # Ưu tiên dòng MobileCLIP + DINOv2 Re-ranking nếu có
            if "Method" in dfm.columns:
                mask = dfm["Method"].astype(str).str.contains("DINO", case=False, na=False)
                row = dfm[mask].iloc[0] if mask.any() else dfm.iloc[-1]
            else:
                row = dfm.iloc[-1]

            if "Alpha" in dfm.columns and pd.notna(row.get("Alpha")):
                alpha = float(row["Alpha"])
            elif "Alpha tối ưu (Validation)" in dfm.columns and pd.notna(row.get("Alpha tối ưu (Validation)")):
                alpha = float(row["Alpha tối ưu (Validation)"])

            if "Retrieval_K" in dfm.columns and str(row.get("Retrieval_K")) not in ["-", "nan", "None"]:
                retrieval_k = int(float(row["Retrieval_K"]))
        except Exception:
            pass
    return {"alpha": alpha, "retrieval_k": retrieval_k, "beta": beta, "metric_csv": metric_csv}

# ============================================================
# 5. LOAD MODEL GIỐNG PIPELINE
# ============================================================

@st.cache_resource(show_spinner="⏳ Đang load MobileCLIP/CLIP model...")
def load_clip_model():
    """Giống pipeline: thử MobileCLIP Apple trước, lỗi thì fallback HuggingFace CLIP."""
    try:
        import mobileclip
        import urllib.request

        ckpt_urls = {
            "mobileclip_s0": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt",
            "mobileclip_s1": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s1.pt",
            "mobileclip_s2": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt",
            "mobileclip_b": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_b.pt",
        }
        if not os.path.exists(MOBILECLIP_CKPT):
            urllib.request.urlretrieve(ckpt_urls[MOBILECLIP_VARIANT], MOBILECLIP_CKPT)

        model, _, preprocess = mobileclip.create_model_and_transforms(
            MOBILECLIP_VARIANT, pretrained=MOBILECLIP_CKPT
        )
        tokenizer = mobileclip.get_tokenizer(MOBILECLIP_VARIANT)
        model = model.to(DEVICE).eval()
        return {
            "backend": "mobileclip",
            "label": f"MobileCLIP ({MOBILECLIP_VARIANT})",
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
        }
    except Exception:
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(HF_MODEL).to(DEVICE).eval()
        processor = CLIPProcessor.from_pretrained(HF_MODEL)
        return {
            "backend": "hf_clip",
            "label": f"CLIP ({HF_MODEL})",
            "model": model,
            "preprocess": processor,
            "tokenizer": None,
        }


@torch.no_grad()
def extract_query_clip_features(img_path: str, title: str, clip_pack: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    model = clip_pack["model"]
    backend = clip_pack["backend"]
    preprocess = clip_pack["preprocess"]
    tokenizer = clip_pack["tokenizer"]

    img = Image.open(img_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    title = title or ""

    if backend == "mobileclip":
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        img_feat = model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        tokens = tokenizer([title]).to(DEVICE)
        txt_feat = model.encode_text(tokens)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        return img_feat.cpu().float().numpy()[0], txt_feat.cpu().float().numpy()[0]

    inputs_img = preprocess(images=[img], return_tensors="pt", padding=True).to(DEVICE)
    img_feat = model.get_image_features(**inputs_img)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    inputs_txt = preprocess(text=[title], return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEVICE)
    txt_feat = model.get_text_features(**inputs_txt)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    return img_feat.cpu().float().numpy()[0], txt_feat.cpu().float().numpy()[0]


@st.cache_resource(show_spinner="⏳ Đang load DINOv2 model...")
def load_dino_model():
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dinov2 = dinov2.to(DEVICE).eval()
    transform_dino = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return dinov2, transform_dino


@torch.no_grad()
def get_dinov2_embedding(img_path: str, dino_pack) -> np.ndarray:
    dinov2, transform_dino = dino_pack
    try:
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.exif_transpose(img)
    except Exception:
        img = Image.new("RGB", (224, 224), (128, 128, 128))
    img_tensor = transform_dino(img).unsqueeze(0).to(DEVICE)
    feats = dinov2.forward_features(img_tensor)["x_norm_clstoken"]
    feats = F.normalize(feats, dim=-1)
    return feats.cpu().numpy().flatten().astype(np.float32)

# ============================================================
# 6. SEARCH ENGINE: ĐÚNG LOGIC search_two_stage
# ============================================================

def fuse_and_normalize_clip(img_feats: np.ndarray, txt_feats: np.ndarray, alpha: float) -> np.ndarray:
    fused = alpha * img_feats + (1 - alpha) * txt_feats
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return (fused / norms).astype(np.float32)


def fuse_single_clip(img_feat: np.ndarray, txt_feat: np.ndarray, alpha: float) -> np.ndarray:
    fused = alpha * img_feat + (1 - alpha) * txt_feat
    norm = np.linalg.norm(fused)
    if norm < 1e-10:
        norm = 1e-10
    return (fused / norm).astype(np.float32)


@st.cache_resource(show_spinner="⏳ Đang load feature cache và build FAISS index...")
def load_search_engine():
    if faiss is None:
        raise ImportError("Thiếu faiss. Cài bằng: pip install faiss-cpu")

    data_dir = find_data_dir()
    image_dir = find_image_dir(data_dir)
    feature_dir = find_feature_dir()
    params = load_pipeline_params()

    train_csv = data_dir / "train.csv"
    df_gallery = pd.read_csv(train_csv).reset_index(drop=True)
    if "image" not in df_gallery.columns:
        raise ValueError("train.csv thiếu cột image.")
    if "posting_id" not in df_gallery.columns:
        df_gallery["posting_id"] = df_gallery.index.astype(str)
    if "label_group" not in df_gallery.columns:
        df_gallery["label_group"] = "unknown"
    if "title" not in df_gallery.columns:
        df_gallery["title"] = ""

    gallery_img_feats = np.load(feature_dir / "mobileclip_gallery_img.npy").astype(np.float32)
    gallery_txt_feats = np.load(feature_dir / "mobileclip_gallery_txt.npy").astype(np.float32)
    gallery_dino = np.load(feature_dir / "dinov2_gallery.npy").astype(np.float32)

    n = len(df_gallery)
    if gallery_img_feats.shape[0] != n or gallery_txt_feats.shape[0] != n or gallery_dino.shape[0] != n:
        raise ValueError(
            f"Feature cache lệch train.csv: train={n}, "
            f"img={gallery_img_feats.shape[0]}, txt={gallery_txt_feats.shape[0]}, dino={gallery_dino.shape[0]}."
        )

    df_gallery["resolved_image_path"] = df_gallery["image"].astype(str).map(lambda x: resolve_image_path(x, image_dir))
    df_gallery["image_basename"] = df_gallery["image"].astype(str).map(lambda x: Path(x).name)

    alpha = float(params["alpha"])
    retrieval_k = int(params["retrieval_k"])
    beta = float(params["beta"])

    gallery_fused_stage1 = fuse_and_normalize_clip(gallery_img_feats, gallery_txt_feats, alpha)
    faiss_index_stage1 = faiss.IndexFlatIP(gallery_fused_stage1.shape[1])
    faiss_index_stage1.add(gallery_fused_stage1)

    basename_to_idx = {name: i for i, name in enumerate(df_gallery["image_basename"].tolist())}

    return {
        "data_dir": data_dir,
        "image_dir": image_dir,
        "feature_dir": feature_dir,
        "metric_csv": params["metric_csv"],
        "df_gallery": df_gallery,
        "gallery_img_feats": gallery_img_feats,
        "gallery_txt_feats": gallery_txt_feats,
        "gallery_dino": gallery_dino,
        "faiss_index_stage1": faiss_index_stage1,
        "basename_to_idx": basename_to_idx,
        "alpha": alpha,
        "retrieval_k": retrieval_k,
        "beta": beta,
    }


def search_mobileclip_stage1(query_fused_vec: np.ndarray, faiss_index_stage1, top_k: int):
    q = query_fused_vec.reshape(1, -1).astype(np.float32)
    scores, indices = faiss_index_stage1.search(q, top_k)
    return indices[0].tolist(), scores[0].tolist()


def search_two_stage(
    query_img_path: str,
    query_img_feat: np.ndarray,
    query_txt_feat: np.ndarray,
    engine: Dict[str, Any],
    dino_pack,
    query_idx: Optional[int] = None,
    retrieval_k: Optional[int] = None,
    final_k: int = 5,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> pd.DataFrame:
    """Bản Streamlit của hàm search_two_stage trong pipeline chính."""
    df_gallery = engine["df_gallery"]
    gallery_dino = engine["gallery_dino"]
    faiss_index_stage1 = engine["faiss_index_stage1"]

    alpha = engine["alpha"] if alpha is None else float(alpha)
    beta = engine["beta"] if beta is None else float(beta)
    retrieval_k = engine["retrieval_k"] if retrieval_k is None else int(retrieval_k)

    q_fused = fuse_single_clip(query_img_feat, query_txt_feat, alpha)
    raw_indices, raw_scores = search_mobileclip_stage1(
        q_fused, faiss_index_stage1, top_k=min(retrieval_k + 1, len(df_gallery))
    )

    candidate_indices, candidate_clip_scores = [], []
    for idx, score in zip(raw_indices, raw_scores):
        idx = int(idx)
        if idx < 0 or idx >= len(df_gallery):
            continue
        if query_idx is not None and idx == query_idx:
            continue
        candidate_indices.append(idx)
        candidate_clip_scores.append(float(score))
        if len(candidate_indices) == retrieval_k:
            break

    if not candidate_indices:
        raise ValueError("Không tìm được candidate ở GĐ1. Kiểm tra FAISS index và feature cache.")

    q_dino = get_dinov2_embedding(query_img_path, dino_pack)
    dino_scores = [float(np.dot(q_dino, gallery_dino[idx])) for idx in candidate_indices]
    combined = beta * np.array(dino_scores) + (1 - beta) * np.array(candidate_clip_scores)

    order = np.argsort(combined)[::-1][:final_k]

    q_label = None
    q_image = Path(query_img_path).name
    if query_idx is not None:
        q_row = df_gallery.iloc[query_idx]
        q_label = q_row.get("label_group", "unknown")
        q_image = q_row.get("image", q_image)

    rows = []
    for rank, pos in enumerate(order, start=1):
        idx = candidate_indices[int(pos)]
        r = df_gallery.iloc[idx]
        is_correct = None if q_label is None else bool(q_label == r.get("label_group", "unknown"))
        rows.append({
            "rank": rank,
            "method": "MobileCLIP/CLIP + DINOv2 re-ranking",
            "result_idx": idx,
            "query_idx": query_idx,
            "query_image": str(q_image),
            "result_image": str(r["image"]),
            "query_label": "external_unknown" if q_label is None else q_label,
            "result_label": r.get("label_group", "unknown"),
            "is_correct": is_correct,
            "clip_score": float(candidate_clip_scores[int(pos)]),
            "dino_score": float(dino_scores[int(pos)]),
            "final_score": float(combined[int(pos)]),
            "result_path": str(r["resolved_image_path"]),
            "alpha": alpha,
            "beta": beta,
            "retrieval_k": retrieval_k,
        })
    return pd.DataFrame(rows)

# ============================================================
# 7. LOAD ENGINE
# ============================================================

try:
    engine = load_search_engine()
    clip_pack = load_clip_model()
    dino_pack = load_dino_model()
    engine_error = None
except Exception as exc:
    engine = None
    clip_pack = None
    dino_pack = None
    engine_error = exc

# ============================================================
# 8. UI
# ============================================================

if engine_error is not None:
    st.error(f"❌ Không khởi tạo được app: {engine_error}")
    st.info(
        "App này cần đúng output của pipeline chính: train.csv, train_images, "
        "mobileclip_gallery_img.npy, mobileclip_gallery_txt.npy, dinov2_gallery.npy. "
        "Đừng đưa top_indices vào nữa, bản này không dùng thứ đó."
    )
    st.stop()

with st.expander("⚙️ Thông tin pipeline đang dùng", expanded=False):
    st.write(f"**Model query:** {clip_pack['label']}")
    st.write(f"**Thiết bị:** {DEVICE}")
    st.write(f"**Data dir:** `{engine['data_dir']}`")
    st.write(f"**Image dir:** `{engine['image_dir']}`")
    st.write(f"**Feature dir:** `{engine['feature_dir']}`")
    st.write(f"**Metric CSV:** `{engine['metric_csv']}`")
    st.write(f"**Alpha:** `{engine['alpha']}` | **Beta:** `{engine['beta']}` | **Retrieval_K:** `{engine['retrieval_k']}`")

col_input, col_output = st.columns([1, 2.4], gap="large")

with col_input:
    st.markdown('<p class="section-label">📤 Ảnh truy vấn</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Kéo thả hoặc chọn ảnh sản phẩm",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    query_title = st.text_input(
        "Tiêu đề sản phẩm nếu có, bỏ trống nếu chỉ dùng ảnh",
        value="",
        help="Pipeline chính dùng image + title. Với ảnh ngoài dataset, app không biết title nên em có thể nhập tay. Không nhập thì text rỗng.",
    )

    if uploaded_file is not None:
        query_img = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="query-card">', unsafe_allow_html=True)
        st.image(query_img, use_container_width=True)
        st.markdown(f'<p class="img-name">📌 {uploaded_file.name}</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("🖼️ Chưa có ảnh nào được chọn", icon="ℹ️")

    st.markdown("---")
    search_clicked = st.button("🔍 Tìm kiếm sản phẩm tương đồng", use_container_width=True)

with col_output:
    st.markdown('<p class="section-label">🏆 Top 5 kết quả tương đồng</p>', unsafe_allow_html=True)

    if not search_clicked:
        st.markdown(
            """
            <div style="text-align:center;padding:60px 0;color:#ccc;">
                <div style="font-size:3rem">🛍️</div>
                <p style="font-size:1rem;margin-top:12px;">Tải ảnh lên rồi bấm tìm kiếm. Máy móc sẽ làm phần cực nhọc thay con người.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if uploaded_file is None:
            st.warning("⚠️ Upload ảnh trước đã. Không có ảnh mà đòi tìm ảnh thì hơi siêu hình.", icon="⚠️")
            st.stop()

        query_tmp_path = None
        try:
            suffix = Path(uploaded_file.name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempfile.gettempdir()) as tmp:
                tmp.write(uploaded_file.getvalue())
                query_tmp_path = tmp.name

            # Nếu ảnh upload trùng tên với ảnh trong train.csv, dùng feature text/image cache đúng như pipeline đánh giá.
            basename = Path(uploaded_file.name).name
            query_idx = engine["basename_to_idx"].get(basename)
            if query_idx is not None:
                query_img_feat = engine["gallery_img_feats"][query_idx]
                query_txt_feat = engine["gallery_txt_feats"][query_idx]
                match_mode = "dataset_cache_match"
            else:
                query_img_feat, query_txt_feat = extract_query_clip_features(query_tmp_path, query_title, clip_pack)
                match_mode = "external_query_inference"

            top_k_df = search_two_stage(
                query_img_path=query_tmp_path,
                query_img_feat=query_img_feat,
                query_txt_feat=query_txt_feat,
                engine=engine,
                dino_pack=dino_pack,
                query_idx=query_idx,
                retrieval_k=engine["retrieval_k"],
                final_k=FINAL_K,
                alpha=engine["alpha"],
                beta=engine["beta"],
            )
            top_k_df["match_mode"] = match_mode

        except Exception as exc:
            st.error(f"❌ Lỗi khi tìm kiếm: {exc}")
            st.stop()
        finally:
            if query_tmp_path and os.path.exists(query_tmp_path):
                try:
                    os.unlink(query_tmp_path)
                except Exception:
                    pass

        st.success(
            f"✅ Tìm thấy {len(top_k_df)} kết quả bằng MobileCLIP/CLIP + DINOv2 re-ranking "
            f"(mode: {top_k_df.iloc[0]['match_mode']})",
            icon="🎯",
        )

        result_cols = st.columns(len(top_k_df), gap="small")
        for col_ui, (_, row) in zip(result_cols, top_k_df.iterrows()):
            with col_ui:
                result_path = Path(str(row.get("result_path", "")))
                img_display = None
                if result_path.exists():
                    try:
                        img_display = Image.open(result_path).convert("RGB")
                    except Exception:
                        img_display = None

                is_correct = row.get("is_correct", None)
                if is_correct is True:
                    correctness_html = '<span class="correct-tag">✅ Đúng</span>'
                elif is_correct is False:
                    correctness_html = '<span class="wrong-tag">❌ Sai</span>'
                else:
                    correctness_html = '<span style="color:#aaa">Ảnh ngoài dataset</span>'

                result_name = str(row.get("result_image", ""))
                short_name = result_name if len(result_name) <= 28 else result_name[:13] + "…" + result_name[-10:]

                st.markdown(f'<div class="result-card"><span class="rank-badge">TOP {int(row["rank"])}</span>', unsafe_allow_html=True)
                if img_display is not None:
                    st.image(img_display, use_container_width=True)
                else:
                    st.markdown(
                        '<div style="height:140px;background:#f5f5f5;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#bbb;font-size:2rem;">🖼️</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'{correctness_html}<br>'
                    f'<span class="score-badge">score: {float(row["final_score"]):.4f}</span>'
                    f'<p class="img-name">{short_name}</p></div>',
                    unsafe_allow_html=True,
                )

        with st.expander("📊 Xem bảng kết quả chi tiết"):
            display_cols = [
                "rank", "result_image", "result_label", "is_correct", "method", "match_mode",
                "clip_score", "dino_score", "final_score", "alpha", "beta", "retrieval_k", "result_path",
            ]
            st.dataframe(top_k_df[display_cols], use_container_width=True)

st.markdown(
    """
    <hr style="border:1px solid #f0e8e6;margin-top:40px"/>
    <p style="text-align:center;font-size:0.78rem;color:#ccc;padding-bottom:8px">
        Shopee Visual Search Demo · Khớp pipeline chính: MobileCLIP/CLIP + FAISS + DINOv2 Re-ranking
    </p>
    """,
    unsafe_allow_html=True,
)
