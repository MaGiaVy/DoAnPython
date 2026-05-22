Bạn là một Chuyên gia Trí tuệ Nhân tạo và Lập trình viên Python cấp cao. Nhiệm vụ của bạn là viết code hoàn chỉnh cho một Jupyter Notebook chạy trên Google Colab (GPU T4) để giải quyết bài toán "Visual Search trên Shopee".

Đây là Phương pháp chính của nhóm Tuần 4: HỌC ĐA PHƯƠNG THỨC THỰC DỤNG (Multimodal Late Fusion: EfficientNet-B4 + TF-IDF + FAISS).

Vui lòng sinh ra mã nguồn cho 9 CELL theo đúng các QUY TẮC SỐNG CÒN và YÊU CẦU CHI TIẾT dưới đây. Code sinh ra phải sạch, tối ưu (vectorization), có thanh tiến trình `tqdm` và chạy được ngay lập tức không báo lỗi.

### I. QUY TẮC SỐNG CÒN (TUYỆT ĐỐI TUÂN THỦ):
1. KHÔNG DÙNG `.head()`: Bắt buộc chạy trên toàn bộ tập dữ liệu gốc `train.csv` (34.250 ảnh). Không được cắt nhỏ dữ liệu.
2. BẢN CHẤT MULTIMODAL MỚI:
   - Ảnh (Image): Dùng `torchvision.models.efficientnet_b4` (pretrained). Bỏ lớp phân loại cuối cùng để lấy vector đặc trưng (1792 chiều).
   - Chữ (Text): BẮT BUỘC dùng `TfidfVectorizer` từ `sklearn`. Phải có hàm làm sạch text (xóa ký tự đặc biệt, lowercase). Thiết lập `max_features=3000` để cân bằng không gian vector với hình ảnh, chuyển output thưa (sparse) sang dạng dày (dense tensor).
3. THUẬT TOÁN LATE FUSION: Chuẩn hóa L2-Norm cho vector EfficientNet -> Chuẩn hóa L2-Norm cho vector TF-IDF -> Nối ngang (Concatenate) thành siêu vector (1792 + 3000 = 4792 chiều) -> Chuẩn hóa L2-Norm siêu vector này một lần nữa.
4. TÌM KIẾM BẰNG FAISS: Sử dụng `faiss.IndexFlatIP` (vì vector đã L2-norm nên Inner Product sẽ tương đương Cosine Similarity). 
5. CƠ CHẾ SELF-SIMILARITY REMOVAL: Khi truy vấn Top-K, hệ thống BẮT BUỘC phải tự động loại bỏ chính ảnh Query ra khỏi danh sách kết quả.

---

### II. CẤU TRÚC 9 CELL CẦN VIẾT:

**CELL 1: Import thư viện**
- Nạp pandas, numpy, torch, torchvision, faiss, PIL, matplotlib, seaborn, tqdm.
- Nạp `TfidfVectorizer` từ `sklearn.feature_extraction.text`.
- Thêm lệnh `!pip install faiss-gpu` ở đầu cell.

**CELL 2: Cấu hình đường dẫn**
- Mount Google Drive. Tạo các biến đường dẫn: `DATA_DIR` (chứa train.csv, train_images), `PROCESSED` (chứa features), `RESULTS`. Tự tạo thư mục nếu chưa có.

**CELL 3: Load dữ liệu chung**
- Load `train.csv`. Chỉ giữ lại các ảnh thuộc `label_group` có từ 2 ảnh trở lên.
- Lưu lại thành `candidate_df_chung.csv` vào thư mục PROCESSED để dùng chung.

**CELL 4: Trích xuất EfficientNet-B4 (Ảnh)**
- Dùng `try-except` khi đọc ảnh bằng PIL để tránh lỗi corrupt.
- Transform: Resize 380x380 (chuẩn của B4), Center Crop, ToTensor, Normalize(ImageNet).
- Chạy qua model lấy vector 1792 chiều. Lưu vector ra `efficientnet_b4_features.npy`. Nếu file đã có, load trực tiếp.

**CELL 5: Trích xuất TF-IDF (Văn bản - Title)**
- Viết hàm `clean_text` (regex bỏ ký tự đặc biệt, dấu câu, lowercase).
- Khởi tạo `TfidfVectorizer(max_features=3000)`. Fit_transform cột `title` đã clean.
- Convert ma trận kết quả sang NumPy array (dense). Lưu vector ra `tfidf_features.npy`. Nếu đã có, load trực tiếp.

**CELL 6: Tích hợp Late Fusion & FAISS**
- Áp dụng đúng quy tắc Late Fusion ở trên (L2 Norm -> Concat -> L2 Norm).
- Chuyển siêu vector (4792 chiều) thành float32, nạp vào `faiss.IndexFlatIP` (đưa lên GPU nếu có).
- Lưu siêu vector ra `fusion_eff_tfidf_features.npy`.

**CELL 7: Tính Metric Thực nghiệm**
- Duyệt qua từng ảnh làm Query. Dùng FAISS tìm MAX_K + 1 (K_LIST = [1, 3, 5, 10]). 
- Bắt buộc loại bỏ Query gốc khỏi kết quả bằng cách check ID. Tính mAP@5, Precision@[1,3,5,10], Recall@[1,3,5,10].
- In kết quả ra màn hình. Lưu 2 file CSV vào thư mục RESULTS: `metrics_fusion_eff_tfidf.csv` và `fusion_detail_eff_tfidf.csv`.

**CELL 8: Vẽ biểu đồ Phân tích lỗi (Error Analysis)**
- Viết hàm matplotlib vẽ trực quan 1 ảnh Query và 5 ảnh Top-5 trả về. In rõ `label_group` và `title` (cắt ngắn) trên từng ảnh. Ảnh đúng viền xanh, ảnh sai viền đỏ.
- Code tự động quét và vẽ ra: 2 mẫu Đúng Hoàn Toàn (True Positives) và 2 mẫu Sai Top-1 (False Positives). Lưu hình ra thư mục RESULTS.

**CELL 9: Khung Nhận xét (Markdown)**
- Tạo một cell Markdown định dạng đẹp, gạch đầu dòng 5 nhận xét chuyên sâu về: (1) Sự vượt trội của EfficientNet trong việc phân loại tinh (fine-grained), (2) Sự thực dụng của TF-IDF khi vượt qua rào cản đa ngôn ngữ và spam từ khóa của Shopee, (3) Lợi ích của FAISS trong việc tăng tốc độ quét siêu vector 4792 chiều, (4) Phân tích lý do các mẫu False Positives còn sót lại, (5) Hướng tinh chỉnh (Fine-tuning) bằng ArcFace cho Tuần 5.

Hãy viết code thật sạch sẽ, không dùng placeholder, đảm bảo có thể Copy-Paste và Run All thành công ngay lập tức trên Google Colab!