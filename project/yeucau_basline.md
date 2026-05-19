# YÊU CẦU DỰ ÁN: XÂY DỰNG 2 BASELINE BỔ SUNG (SHOPEE VISUAL SEARCH)

**Vai trò của bạn:** Bạn là một Senior Data Scientist. Nhiệm vụ của bạn là viết code cho Jupyter Notebook (`.ipynb`) để triển khai thêm 2 mô hình Baseline cho bài toán truy xuất sản phẩm tương đồng của Shopee.

## I. MÔ TẢ CÔNG VIỆC (2 TASKS)

Tập dữ liệu đang sử dụng là `train.csv` (có 34,250 dòng). Tập đánh giá (evaluation) là một subset gồm 500 ảnh đã được định nghĩa từ trước. Bạn cần viết code thực hiện 2 Baseline sau trên tập 500 mẫu này:

### Task 1: Baseline 2 - Truy xuất bằng mã băm hình ảnh (Perceptual Hash)
* **Dữ liệu:** Sử dụng cột `image_phash` (chứa các mã băm dạng chuỗi hex, ví dụ: `94974f937d4c2433`).
* **Phương pháp:** * Tính khoảng cách Hamming (Hamming distance) giữa các chuỗi pHash của ảnh truy vấn (query) và tập gallery (500 ảnh).
    * Khoảng cách Hamming càng nhỏ -> Điểm tương đồng càng cao.
    * Truy xuất Top-K ảnh có khoảng cách Hamming thấp nhất (loại bỏ ảnh trùng với chính query).
* **Đánh giá:** Tính Precision@K, Recall@K (với K=1, 3, 5, 10) và mAP.

### Task 2: Baseline 3 - Truy xuất bằng văn bản (TF-IDF trên Tiêu đề)
* **Dữ liệu:** Sử dụng cột `title` (chứa tên sản phẩm).
* **Phương pháp:**
    * Sử dụng `TfidfVectorizer` từ `sklearn.feature_extraction.text` để chuyển đổi toàn bộ `title` trong tập 500 mẫu thành các vector đặc trưng (Text Embeddings).
    * Tính khoảng cách Cosine Similarity giữa vector tiêu đề truy vấn và các vector trong gallery.
    * Truy xuất Top-K sản phẩm có điểm Cosine cao nhất (loại bỏ chính nó).
* **Đánh giá:** Tính Precision@K, Recall@K (với K=1, 3, 5, 10) và mAP.

---

## II. QUY TẮC CODE & TRÌNH BÀY (BẮT BUỘC TUÂN THỦ)

1.  **Cấu trúc Notebook:** Tuân thủ nghiêm ngặt mô hình xen kẽ: 
    * `Markdown` (Giải thích mục đích của Cell code sắp tới).
    * `Code` (Triển khai logic).
    * `Markdown` (In đậm nhận xét kết quả thu được, phân tích tại sao điểm cao/thấp).
2.  **Định dạng Print (f-string):** Mọi lệnh `print` phải dùng f-string. 
    * Số lượng lớn phải có dấu phẩy hàng ngàn: `{value:,}`.
    * Tỷ lệ % hoặc điểm metric (mAP, Recall) phải làm tròn 4 chữ số thập phân: `{score:.4f}`.
3.  **Lưu trữ Kết quả:** * Nếu có vẽ biểu đồ so sánh mAP/Recall giữa các Baseline, sử dụng `matplotlib`/`seaborn`.
    * Lưu biểu đồ vào `../results/` với đuôi `.png`, định dạng `dpi=150`, `bbox_inches='tight'`.
4.  **Hành văn & Nhận xét:** * Sử dụng tiếng Việt học thuật, rành mạch.
    * Cuối notebook, phải có một bảng tổng hợp so sánh (bằng Pandas DataFrame) giữa: Baseline 1 (ResNet50 - đã làm), Baseline 2 (pHash) và Baseline 3 (TF-IDF). Đưa ra kết luận phương pháp nào đang hiệu quả nhất trên tập dữ liệu này.