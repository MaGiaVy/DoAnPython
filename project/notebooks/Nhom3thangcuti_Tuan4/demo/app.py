# ============================================================
# app.py  –  Visual Search Demo (Shopee Product Image Retrieval)
# Framework : Streamlit
# Tác giả   : Nhóm 3 – Tuần 4
#
# Cách chạy (cd vào thư mục chứa app.py trước):
#   cd notebooks/Nhom3thangcuti_Tuan4/demo
#   streamlit run app.py
#   streamlit run notebooks\Nhom3thangcuti_Tuan4\demo\app.py
# ============================================================

import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# ============================================================
# ⚙️  CẤU HÌNH TRANG  (phải gọi đầu tiên, trước mọi lệnh st.)
# ============================================================
st.set_page_config(
    page_title="Shopee Visual Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 🎨  CUSTOM CSS  – Giao diện mang màu sắc Shopee
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }

    .hero-banner {
        background: linear-gradient(135deg, #EE4D2D 0%, #FF7337 100%);
        border-radius: 16px; padding: 32px 40px; margin-bottom: 28px;
        display: flex; align-items: center; gap: 18px;
        box-shadow: 0 8px 30px rgba(238,77,45,0.35);
    }
    .hero-title { font-size: 2rem; font-weight: 700; color: #fff; line-height: 1.2; margin: 0; }
    .hero-sub   { font-size: 0.95rem; color: rgba(255,255,255,0.88); margin: 6px 0 0; }
    .hero-icon  { font-size: 3rem; }

    .section-label {
        font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: #EE4D2D; margin-bottom: 10px;
    }
    .query-card {
        background: #fff8f6; border: 2px solid #ffd5c8;
        border-radius: 14px; padding: 18px; text-align: center;
    }
    .query-title { font-size: 0.85rem; font-weight: 600; color: #EE4D2D; margin-top: 10px; }

    .result-card {
        background: #ffffff; border: 1.5px solid #f0e8e6;
        border-radius: 12px; padding: 12px; text-align: center;
        transition: box-shadow .2s; height: 100%;
    }
    .result-card:hover { box-shadow: 0 6px 24px rgba(238,77,45,0.18); }

    .rank-badge {
        display: inline-block; background: #EE4D2D; color: #fff;
        font-size: 0.72rem; font-weight: 700; border-radius: 20px;
        padding: 2px 10px; margin-bottom: 8px;
    }
    .img-name   { font-size: 0.73rem; color: #666; word-break: break-all; margin-top: 6px; }
    .score-badge {
        display: inline-block; background: #fff3f0; color: #EE4D2D;
        border: 1px solid #ffcbbf; font-size: 0.72rem; font-weight: 600;
        border-radius: 20px; padding: 2px 10px; margin-top: 4px;
    }
    .correct-tag { color: #27ae60; font-weight: 700; }
    .wrong-tag   { color: #e74c3c; font-weight: 700; }

    div.stButton > button {
        background: linear-gradient(135deg, #EE4D2D, #FF7337) !important;
        color: #fff !important; font-family: 'Be Vietnam Pro', sans-serif !important;
        font-weight: 700 !important; font-size: 1rem !important;
        border: none !important; border-radius: 10px !important;
        padding: 0.6rem 2.2rem !important;
        box-shadow: 0 4px 14px rgba(238,77,45,0.35) !important; width: 100%;
    }
    div.stButton > button:hover  { opacity: .88 !important; transform: translateY(-1px) !important; }
    div.stButton > button:active { transform: translateY(0px) !important; }

    [data-testid="stFileUploader"] {
        border: 2px dashed #ffbfaf !important; border-radius: 12px !important;
        background: #fff8f6 !important; padding: 12px !important;
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 🖼️  HEADER BANNER
# ============================================================
st.markdown(
    """
    <div class="hero-banner">
        <span class="hero-icon">🔍</span>
        <div>
            <p class="hero-title">Shopee Visual Search</p>
            <p class="hero-sub">Truy xuất ảnh sản phẩm tương đồng · Strong Fusion Pipeline · Tuần 4</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# ══════════════════════════════════════════════════════════════
# YÊU CẦU 1 – @st.cache_resource: load index, candidate_df,
#             và toàn bộ logic search() MỘT LẦN DUY NHẤT
# ══════════════════════════════════════════════════════════════
# Hàm này sao chép nguyên si logic tìm file + load dữ liệu từ
# demo.ipynb (phần CONFIG → load → chuẩn hóa → define search).
# Nhờ @st.cache_resource, Streamlit giữ kết quả trong bộ nhớ
# server: dù người dùng upload bao nhiêu ảnh, npy/csv chỉ đọc
# đúng 1 lần lúc khởi động → không bị lag mỗi lần click.
# ============================================================

@st.cache_resource(show_spinner="⏳ Đang tải index & candidate_df lần đầu …")
def load_search_engine():
    """
    Trả về (search_fn, index, candidate_df, IMAGE_DIR).

    ┌─────────────────────────────────────────────────────────┐
    │  CẤU HÌNH – chỉnh 3 dòng MANUAL_* nếu tự-tìm thất bại │
    └─────────────────────────────────────────────────────────┘
    """
    # ── A. CONFIG (giữ nguyên như demo.ipynb) ──────────────────
    MANUAL_TOP_NPY       = r"E:\study\DoAnPython\project\data\processed\STRONG_best_top_indices_xxx.npy"
    MANUAL_CANDIDATE_CSV = r"E:\study\DoAnPython\project\data\processed\candidate_df_strong_yolo_crop_fusion.csv"
    MANUAL_IMAGE_DIR     = r"E:\study\DuLieuPython\data\raw\train_images"
    ALLOW_PHASH_FALLBACK = True

    # ── B. HELPER: tìm file tự động (copy từ demo.ipynb) ───────
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
            key = str(d.resolve()) if True else str(d)
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out

    # app.py nằm cùng thư mục demo/ → PROJECT_ROOT = thư mục đó
    PROJECT_ROOT   = Path(__file__).parent.resolve()
    ROOT_CANDIDATES = [PROJECT_ROOT] + list(PROJECT_ROOT.parents[:6])

    SEARCH_DIRS = []
    for root in ROOT_CANDIDATES:
        SEARCH_DIRS += [
            root,
            root / "results",
            root / "data" / "processed",
            root / "outputs",
            root / "outputs_twostage_gpu",
            root / "outputs" / "tuan4_two_stage",
        ]
    SEARCH_DIRS = unique_existing_dirs(SEARCH_DIRS)

    def find_latest(patterns, manual_path="", required=True, name="file") -> Optional[Path]:
        manual = existing_path(manual_path)
        if manual is not None:
            return manual
        matches = []
        for base in SEARCH_DIRS:
            for pat in patterns:
                try:
                    matches.extend(p for p in base.glob(pat) if p.is_file())
                except Exception:
                    pass
        uniq = {}
        for p in matches:
            uniq[str(p.resolve())] = p
        matches = sorted(uniq.values(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            if required:
                raise FileNotFoundError(f"Không tìm thấy {name}. Hãy sửa MANUAL_* trong load_search_engine().")
            return None
        return matches[0]

    def find_image_dir(manual_path="") -> Optional[Path]:
        manual = existing_path(manual_path)
        if manual and manual.is_dir():
            return manual
        candidates = []
        for root in ROOT_CANDIDATES:
            candidates += [
                root / "data" / "raw" / "train_images",
                root / "project" / "data" / "raw" / "train_images",
            ]
        for c in candidates:
            if c.exists() and c.is_dir():
                return c
        return None

    # ── C. TÌM VÀ LOAD FILE ─────────────────────────────────────
    TOP_NPY = find_latest(
        patterns=[
            "**/STRONG_best_top_indices_*.npy",
            "**/FINAL_fusion_resnet_tfidf_top_indices_*.npy",
            "**/*top_indices*.npy",
            "**/*top_idx*.npy",
        ],
        manual_path=MANUAL_TOP_NPY,
        required=True,
        name="top_indices NPY",
    )
    CANDIDATE_CSV = find_latest(
        patterns=[
            "**/candidate_df_strong_yolo_crop_fusion.csv",
            "**/candidate_df_twostage_yolo_sahi_effb4_tfidf.csv",
            "**/candidate_df_resnet_tfidf_fusion.csv",
            "**/candidate_df*.csv",
        ],
        manual_path=MANUAL_CANDIDATE_CSV,
        required=True,
        name="candidate_df CSV",
    )
    IMAGE_DIR = find_image_dir(MANUAL_IMAGE_DIR)

    # ── D. LOAD VÀ CHUẨN HÓA DỮ LIỆU ───────────────────────────
    candidate_df = pd.read_csv(CANDIDATE_CSV)
    index        = np.load(TOP_NPY)           # shape (N, K)

    # Đảm bảo cột "image" luôn tồn tại
    if "image" not in candidate_df.columns:
        if "image_name" in candidate_df.columns:
            candidate_df["image"] = candidate_df["image_name"]
        elif "posting_id" in candidate_df.columns:
            candidate_df["image"] = candidate_df["posting_id"].astype(str)
        else:
            raise ValueError("candidate_df thiếu cột image / image_name / posting_id.")

    if "label_group" not in candidate_df.columns:
        candidate_df["label_group"] = "unknown"

    if index.shape[0] != len(candidate_df):
        raise ValueError(
            f"top_indices ({index.shape[0]} dòng) lệch candidate_df ({len(candidate_df)} dòng)."
        )

    # Thêm cột resolved_image_path và image_basename để search() dùng
    def resolve_image_path_from_row(row) -> str:
        # Ưu tiên 1: image_path có sẵn trong CSV và tồn tại trên đĩa
        if "image_path" in row and pd.notna(row["image_path"]):
            p = Path(str(row["image_path"]))
            if p.exists():
                return str(p)
        # Ưu tiên 2: IMAGE_DIR / tên file
        name = str(row["image"])
        candidates = []
        if IMAGE_DIR is not None:
            candidates += [IMAGE_DIR / name, IMAGE_DIR / Path(name).name]
        for root in ROOT_CANDIDATES:
            candidates += [
                root / name,
                root / "data" / "raw" / "train_images" / Path(name).name,
            ]
        for p in candidates:
            if p.exists():
                return str(p)
        # Fallback: trả về tên file thô (render ảnh sẽ fail gracefully)
        return str(row.get("image_path", name))

    candidate_df["resolved_image_path"] = candidate_df.apply(resolve_image_path_from_row, axis=1)
    candidate_df["image_basename"]       = candidate_df["image"].astype(str).map(lambda x: Path(x).name)

    # ── E. ĐỊNH NGHĨA HÀM TIỆN ÍCH (copy từ demo.ipynb) ─────────
    def load_rgb(path, size: Optional[Tuple[int, int]] = None) -> Image.Image:
        img = Image.open(path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        if size:
            img = img.resize(size)
        return img

    def phash_hex(path, hash_size: int = 8, highfreq_factor: int = 4) -> str:
        img_size = hash_size * highfreq_factor
        img = load_rgb(path, size=(img_size, img_size)).convert("L")
        pixels = np.asarray(img, dtype=np.float32)
        n = img_size
        x = np.arange(n)
        u = np.arange(n).reshape(-1, 1)
        basis = np.cos(((2 * x + 1) * u * np.pi) / (2 * n))
        dct = basis @ pixels @ basis.T
        low = dct[:hash_size, :hash_size]
        med = np.median(low.flatten()[1:])
        bits = (low > med).flatten()
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return f"{value:0{hash_size * hash_size // 4}x}"

    def hex_hamming(a, b) -> int:
        try:
            return (int(str(a), 16) ^ int(str(b), 16)).bit_count()
        except Exception:
            return 999

    def resolve_query_path(path_like) -> Path:
        p = Path(path_like)
        candidates = [p, PROJECT_ROOT / p]
        if IMAGE_DIR is not None:
            candidates.append(IMAGE_DIR / p.name)
        for c in candidates:
            if c.exists() and c.is_file():
                return c
        raise FileNotFoundError(f"Không tìm thấy query image: {path_like}")

    def find_query_idx(query_path, max_phash_distance: int = 4):
        q_path = resolve_query_path(query_path)
        q_name = q_path.name
        # Match theo tên file
        matches = np.where(candidate_df["image_basename"].astype(str).values == q_name)[0]
        if len(matches) > 0:
            return int(matches[0]), q_path, "filename_match"
        # Match theo đường dẫn tuyệt đối
        if "resolved_image_path" in candidate_df.columns:
            q_abs = str(q_path.resolve()).lower()
            paths = candidate_df["resolved_image_path"].astype(str).map(lambda x: str(Path(x)).lower())
            matches = np.where(paths.values == q_abs)[0]
            if len(matches) > 0:
                return int(matches[0]), q_path, "path_match"
        # Match pHash
        if "image_phash" in candidate_df.columns:
            q_hash = phash_hex(q_path)
            dists = candidate_df["image_phash"].astype(str).map(lambda h: hex_hamming(q_hash, h)).values
            best_idx = int(np.argmin(dists))
            if int(dists[best_idx]) <= max_phash_distance:
                return best_idx, q_path, f"phash_match_distance_{int(dists[best_idx])}"
        return None, q_path, "external_image_not_in_candidate_df"

    # ── F. HÀM SEARCH CHÍNH (signature đúng như demo.ipynb) ─────
    def search(query_path, index, candidate_df, IMAGE_DIR=None, k: int = 5) -> pd.DataFrame:
        q_idx, q_path, match_mode = find_query_idx(query_path)
        rows = []

        if q_idx is not None:
            # Ảnh thuộc candidate_df → dùng top_indices của Strong Fusion
            q_row = candidate_df.iloc[q_idx]
            seen  = {str(q_row["image_basename"])}
            for idx in list(index[q_idx]):
                idx = int(idx)
                if idx < 0 or idx >= len(candidate_df) or idx == q_idx:
                    continue
                r      = candidate_df.iloc[idx]
                r_name = str(r["image_basename"])
                if r_name in seen:
                    continue
                seen.add(r_name)
                rows.append({
                    "rank"        : len(rows) + 1,
                    "method"      : "main_top_indices",
                    "match_mode"  : match_mode,
                    "query_idx"   : q_idx,
                    "result_idx"  : idx,
                    "query_image" : str(q_row["image"]),
                    "result_image": str(r["image"]),       # tên file ảnh kết quả
                    "query_label" : q_row.get("label_group", "unknown"),
                    "result_label": r.get("label_group", "unknown"),
                    "is_correct"  : bool(q_row.get("label_group") == r.get("label_group")),
                    "query_path"  : str(q_path),
                    "result_path" : str(r["resolved_image_path"]),  # đường dẫn đầy đủ
                })
                if len(rows) >= k:
                    break

        elif ALLOW_PHASH_FALLBACK and "image_phash" in candidate_df.columns:
            # Ảnh ngoài candidate_df → fallback pHash
            q_hash = phash_hex(q_path)
            tmp = candidate_df.copy()
            tmp["phash_distance"] = tmp["image_phash"].astype(str).map(
                lambda h: hex_hamming(q_hash, h)
            )
            for _, r in tmp.sort_values("phash_distance").head(k).iterrows():
                rows.append({
                    "rank"          : len(rows) + 1,
                    "method"        : "fallback_phash_external_query",
                    "match_mode"    : match_mode,
                    "query_idx"     : None,
                    "result_idx"    : int(r.name),
                    "query_image"   : Path(q_path).name,
                    "result_image"  : str(r["image"]),
                    "query_label"   : "external_unknown",
                    "result_label"  : r.get("label_group", "unknown"),
                    "is_correct"    : None,
                    "query_path"    : str(q_path),
                    "result_path"   : str(r["resolved_image_path"]),
                    "phash_distance": int(r["phash_distance"]),
                })
        else:
            raise ValueError(
                "Ảnh query không nằm trong candidate_df và không thể fallback pHash. "
                "Dùng ảnh từ train_images để có kết quả Strong Fusion chuẩn."
            )

        if not rows:
            raise ValueError("Không tạo được Top-K. Kiểm tra index hoặc query image.")

        return pd.DataFrame(rows).head(k)

    # Trả về tất cả những gì app cần
    return search, index, candidate_df, IMAGE_DIR


# ── Gọi 1 lần, Streamlit cache lại cho mọi request tiếp theo ──
search_fn, index, candidate_df, IMAGE_DIR = load_search_engine()

# ============================================================
# 📐  LAYOUT CHÍNH: 2 CỘT (Input trái | Output phải)
# ============================================================
col_input, col_output = st.columns([1, 2.4], gap="large")

# ─────────────────────────────────────────────────────────────
# CỘT TRÁI – Upload ảnh & nút tìm kiếm
# ─────────────────────────────────────────────────────────────
with col_input:
    st.markdown('<p class="section-label">📤 Ảnh truy vấn</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Kéo thả hoặc chọn ảnh sản phẩm",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        query_img = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="query-card">', unsafe_allow_html=True)
        st.image(query_img, use_container_width=True)
        st.markdown(
            f'<p class="query-title">📌 {uploaded_file.name}</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("🖼️ Chưa có ảnh nào được chọn", icon="ℹ️")

    st.markdown("---")
    search_clicked = st.button("🔍  Tìm kiếm sản phẩm tương đồng", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# CỘT PHẢI – Kết quả Top-K
# ─────────────────────────────────────────────────────────────
with col_output:
    st.markdown('<p class="section-label">🏆 Top 5 kết quả tương đồng</p>', unsafe_allow_html=True)

    if not search_clicked:
        st.markdown(
            """
            <div style="text-align:center;padding:60px 0;color:#ccc;">
                <div style="font-size:3rem">🛍️</div>
                <p style="font-size:1rem;margin-top:12px;">
                    Tải lên ảnh sản phẩm và nhấn <b>Tìm kiếm</b> để xem kết quả
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        # ── Bắt lỗi: chưa upload ảnh ────────────────────────────
        if uploaded_file is None:
            st.warning("⚠️ Vui lòng tải lên ảnh truy vấn trước khi tìm kiếm.", icon="⚠️")
            st.stop()

        # ── Bắt lỗi: engine chưa load (file không tìm thấy) ─────
        if search_fn is None:
            st.error(
                "❌ Search engine chưa khởi tạo được. "
                "Kiểm tra MANUAL_* trong hàm load_search_engine() và chạy lại.",
                icon="🛠️",
            )
            st.stop()

        # ════════════════════════════════════════════════════════
        # YÊU CẦU 2 – tempfile: lưu ảnh upload thành file vật lý,
        #             truyền đường dẫn vào search(), xóa sau khi xong
        # ════════════════════════════════════════════════════════
        # Lý do cần file vật lý: hàm search() → resolve_query_path()
        # gọi Path(...).exists() và Image.open(path) – chỉ hoạt động
        # với đường dẫn thật trên đĩa, không nhận BytesIO từ Streamlit.
        # ────────────────────────────────────────────────────────
        query_tmp_path = None
        top_k_df       = None

        with st.spinner("🔄 Đang xử lý ảnh và tìm kiếm …"):
            try:
                # 2a. Tạo file tạm giữ đúng phần mở rộng gốc (.jpg / .png …)
                suffix = Path(uploaded_file.name).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                    dir=tempfile.gettempdir(),
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())   # ghi bytes vào đĩa
                    query_tmp_path = tmp.name             # lưu path để dùng sau

                # 2b. Gọi hàm search() với đường dẫn file tạm
                #     ┌──────────────────────────────────────────────┐
                #     │  search(query_path, index, candidate_df,     │
                #     │         IMAGE_DIR, k=5)                      │
                #     │  query_path  : str, đường dẫn tạm trên đĩa  │
                #     │  index       : np.ndarray (N × K) từ cache   │
                #     │  candidate_df: DataFrame từ cache            │
                #     │  IMAGE_DIR   : Path | None từ cache          │
                #     │  k           : số kết quả cần trả về         │
                #     └──────────────────────────────────────────────┘
                top_k_df = search_fn(
                    query_tmp_path,
                    index,
                    candidate_df,
                    IMAGE_DIR,
                    k=5,
                )

            except FileNotFoundError as exc:
                st.error(f"❌ Không tìm thấy file: {exc}", icon="📁")
                st.stop()
            except ValueError as exc:
                st.error(f"❌ Lỗi dữ liệu: {exc}", icon="⚠️")
                st.stop()
            except Exception as exc:
                st.error(f"❌ Lỗi không xác định: {exc}", icon="🚨")
                st.stop()
            finally:
                # 2c. Xóa file tạm dù search() thành công hay lỗi
                if query_tmp_path and os.path.exists(query_tmp_path):
                    try:
                        os.unlink(query_tmp_path)
                    except Exception:
                        pass

        # ════════════════════════════════════════════════════════
        # YÊU CẦU 3 – Render grid Top-5: nối tên ảnh với IMAGE_DIR
        # ════════════════════════════════════════════════════════
        # Ưu tiên hiển thị: result_path (đường dẫn đầy đủ đã resolve
        # khi load) → nếu không tồn tại, thử nối IMAGE_DIR / result_image
        # → nếu vẫn không có, hiển thị placeholder.
        # ────────────────────────────────────────────────────────
        st.success(
            f"✅ Tìm thấy {len(top_k_df)} kết quả "
            f"(method: {top_k_df.iloc[0].get('method', 'N/A')})",
            icon="🎯",
        )

        result_cols = st.columns(len(top_k_df), gap="small")

        for col_ui, (_, row) in zip(result_cols, top_k_df.iterrows()):
            with col_ui:
                rank = int(row.get("rank", 1))

                # 3a. Xác định đường dẫn ảnh kết quả -------------------
                #   - result_path : đường dẫn đầy đủ (resolved_image_path)
                #                   được search() điền sẵn.
                #   - Nếu result_path không tồn tại trên đĩa máy này,
                #     fallback: IMAGE_DIR / basename(result_image).
                result_path_str = str(row.get("result_path", ""))
                result_img_name = str(row.get("result_image", ""))   # tên file thuần

                img_display: Optional[Image.Image] = None

                # Thử result_path trước (đường dẫn đầy đủ từ DataFrame)
                if result_path_str and Path(result_path_str).exists():
                    try:
                        img_display = Image.open(result_path_str).convert("RGB")
                    except Exception:
                        img_display = None

                # Fallback: IMAGE_DIR / tên file nếu result_path lỗi
                if img_display is None and IMAGE_DIR is not None and result_img_name:
                    fallback_path = Path(IMAGE_DIR) / Path(result_img_name).name
                    if fallback_path.exists():
                        try:
                            img_display = Image.open(str(fallback_path)).convert("RGB")
                        except Exception:
                            img_display = None

                # 3b. Nhãn đúng / sai ──────────────────────────────────
                is_correct = row.get("is_correct", None)
                if is_correct is True:
                    correctness_html = '<span class="correct-tag">✅ Đúng</span>'
                elif is_correct is False:
                    correctness_html = '<span class="wrong-tag">❌ Sai</span>'
                else:
                    correctness_html = '<span style="color:#aaa">❓</span>'

                # 3c. Điểm pHash (chỉ có khi fallback mode) ────────────
                score_html = ""
                if "phash_distance" in row.index and pd.notna(row["phash_distance"]):
                    score_html = (
                        f'<span class="score-badge">'
                        f'pHash dist: {int(row["phash_distance"])}'
                        f'</span>'
                    )

                # 3d. Tên file rút ngắn để hiển thị ────────────────────
                img_name_short = (
                    result_img_name
                    if len(result_img_name) <= 26
                    else result_img_name[:12] + "…" + result_img_name[-10:]
                )

                # 3e. Render card ───────────────────────────────────────
                st.markdown(
                    f'<div class="result-card"><span class="rank-badge">TOP {rank}</span>',
                    unsafe_allow_html=True,
                )

                if img_display is not None:
                    st.image(img_display, use_container_width=True)
                else:
                    st.markdown(
                        '<div style="height:140px;background:#f5f5f5;border-radius:8px;'
                        'display:flex;align-items:center;justify-content:center;'
                        'color:#bbb;font-size:2rem;">🖼️</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"{correctness_html}{score_html}"
                    f'<p class="img-name">{img_name_short}</p></div>',
                    unsafe_allow_html=True,
                )

        # ── Bảng chi tiết mở rộng ───────────────────────────────
        with st.expander("📊 Xem bảng kết quả chi tiết"):
            display_cols = [c for c in [
                "rank", "result_image", "result_label",
                "is_correct", "method", "match_mode",
                "phash_distance", "result_path",
            ] if c in top_k_df.columns]
            st.dataframe(top_k_df[display_cols], use_container_width=True)

# ============================================================
# 📌  FOOTER
# ============================================================
st.markdown(
    """
    <hr style="border:1px solid #f0e8e6;margin-top:40px"/>
    <p style="text-align:center;font-size:0.78rem;color:#ccc;padding-bottom:8px">
        Shopee Visual Search Demo · Nhóm 3 – Tuần 4 ·
        Powered by <b style="color:#EE4D2D">Strong Fusion Pipeline</b>
    </p>
    """,
    unsafe_allow_html=True,
)
