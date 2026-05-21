# QUY TẮC LẬP TRÌNH VÀ CHỈ THỊ VIBECODING: PHƯƠNG PHÁP CHÍNH TUẦN 3

**Dự án:** Hệ thống Tìm kiếm Sản phẩm Tương đồng Shopee (Visual Search)
**Mô hình Đề xuất:** Học đa phương thức Đa mô hình (Multimodal Late Fusion: ResNet50 + CLIP Text + FAISS)
**Môi trường Thực thi:** Google Colab (Yêu cầu tăng tốc phần cứng GPU T4)
**Mục tiêu:** Tự động triển khai cấu trúc mã nguồn hoàn chỉnh gồm 9 cells, đảm bảo xử lý dữ liệu quy mô lớn (34.250 ảnh), tính toán độ đo chính xác và kết xuất tài nguyên phục vụ báo cáo học thuật.

---

## I. NGUYÊN TẮC THỰC THI SỐNG CÒN (MANDATORY RULES)

1. **QUY MÔ TOÀN VẸN (NO HEAD/SUBSET):** Tuyệt đối KHÔNG sử dụng các hàm cắt dữ liệu như `.head()`, `.iloc[:100]` trong các cell tính toán cốt lõi. Toàn bộ pipeline từ trích xuất, nạp chỉ mục đến đánh giá metric ĐỀU PHẢI CHẠY TRÊN TOÀN BỘ DATASET GỐC gồm 34.250 ảnh và 11.014 nhóm (`label_group`).
2. **BẢN CHẤT ĐA PHƯƠNG THỨC (MULTIMODAL):** Phương pháp chính phải sử dụng song song hai luồng thông tin: Luồng ảnh đi qua **ResNet50 Backbone** để lấy đặc trưng thị giác cấp cao; Luồng chữ đi qua **CLIP Text Encoder** để lấy đặc trưng ngữ nghĩa từ cột `title` (Tiêu đề sản phẩm). Tuyệt đối không dùng CLIP mã hóa ảnh trong tệp này để tránh trùng lặp cấu trúc thị giác thuần túy.
3. **CHUẨN HÓA HÌNH HỌC KHÔNG GIAN (L2-NORM):** Trước khi thực hiện phép ghép nối (Concatenation), phải chuẩn hóa L2 cho từng vector nguồn độc lập. Sau khi ghép nối thành siêu vector 2560 chiều, BẮT BUỘC phải áp dụng phép chuẩn hóa L2 một lần nữa để đưa không gian không gian về dạng mặt cầu đơn vị.
4. **CƠ CHẾ LOẠI BỎ NHIỄU TỰ THÂN (SELF-SIMILARITY REMOVAL):** Khi đánh giá hoặc truy xuất Top-K lân cận, hệ thống phải tự động loại bỏ chính thực thể Query ra khỏi danh sách kết quả trả về bằng cách kiểm tra và bỏ qua chỉ số (index) trùng lặp, tránh hiện tượng mô hình "gian lận" độ chính xác.
5. **CHẤT LƯỢNG MÃ NGUỒN:** Code phải sạch, viết theo hướng tối ưu hóa mảng tính toán (Vectorization) bằng NumPy và PyTorch Tensor, hạn chế tối đa các vòng lặp `for` lồng nhau gây nghẽn RAM/GPU. Tất cả các cell phải tích hợp thanh tiến trình `tqdm` để theo dõi tiến độ.

---

## II. CHI TIẾT CẤU TRÚC 9 CELLS CẦN TRIỂN KHAI TRÊN NOTEBOOK

### CELL 1: Khởi tạo và Import Hệ thống Thư viện

Triển khai mã nguồn nạp toàn bộ các gói thư viện cần thiết cho bài toán Học sâu, xử lý ảnh, trích xuất văn bản và tìm kiếm không gian vector siêu tốc.

- **Yêu cầu kỹ thuật:** Thiết lập cấu hình hiển thị tối đa của Pandas, định cấu hình kích thước biểu đồ chuẩn cho Matplotlib (`10x5`) và Seaborn (`whitegrid`). Kiểm tra và in rõ trạng thái khả dụng của GPU CUDA (`Device: cuda` nếu có).
- **Thư viện bắt buộc:** `pandas`, `numpy`, `torch`, `torchvision`, `clip`, `faiss`, `PIL.Image`, `matplotlib.pyplot`, `seaborn`, `os`, `tqdm`.

### CELL 2: Cấu hình Đường dẫn Hệ thống (Google Drive Mount)

Viết mã nguồn kết nối trực tiếp với tài khoản Google Drive để đọc/ghi dữ liệu liên tục, giảm thiểu rủi ro mất mát dữ liệu khi Colab ngắt kết nối.

- **Cấu trúc thư mục quy chuẩn:**
  - `DATA_DIR`: Thư mục chứa dữ liệu thô đầu vào (`train.csv` và thư mục ảnh `train_images`).
  - `PROCESSED`: Thư mục lưu trữ tài nguyên trung gian phục vụ đồng bộ nhóm (`candidate_df_chung.csv`, các tệp đặc trưng `.npy`).
  - `RESULTS`: Thư mục kết xuất thành quả cuối cùng (các biểu đồ ảnh `.png`, bảng điểm `.csv`).
- **Yêu cầu bổ sung:** Tự động kiểm tra tính tồn tại của các đường dẫn cốt lõi (`CSV_PATH`, `IMAGE_DIR`) bằng `os.path.exists()` và in ra ký tự trực quan (✅/❌) để người dùng theo dõi.

### CELL 3: Đồng bộ và Thiết lập Không gian Dữ liệu chung

Nạp tệp dữ liệu dùng chung của cả nhóm nhằm bảo đảm tính công bằng khi đối chứng các phương pháp.

- **Logic thực thi:** Kiểm tra sự tồn tại của tệp `candidate_df_chung.csv` trong thư mục `PROCESSED`.
  - Nếu tồn tại: Tiến hành nạp trực tiếp vào biến `candidate_df`.
  - Nếu chưa tồn tại: Thực hiện lọc sạch tệp `train.csv` gốc, chỉ giữ lại những nhóm nhãn (`label_group`) có tần suất xuất hiện lớn hơn hoặc bằng 2 ảnh, sắp xếp lại chỉ mục và lưu lại vào đường dẫn quy định.
- **Yêu cầu kết xuất văn bản:** In tổng số lượng ảnh hợp lệ, tổng số lượng nhóm nhãn duy nhất để phục vụ kiểm toán số liệu đầu vào.

### CELL 4: Pipeline Trích xuất Đặc trưng Thị giác (ResNet50)

Xây dựng lớp dữ liệu và luồng cấp phát dữ liệu (DataLoader) song song để trích xuất ma trận vector đặc trưng từ hình ảnh sản phẩm.

- **Cơ chế tối ưu:** Kiểm tra sự tồn tại của tệp `resnet50_features_34k.npy`. Nếu đã có, nạp trực tiếp vào bộ nhớ GPU dưới dạng Tensor và bỏ qua bước tính toán lại để tiết kiệm 15-20 phút thời gian biên dịch.
- **Kiến trúc mạng:** Khởi tạo mạng ResNet50 với bộ trọng số ImageNet gốc. Thực hiện loại bỏ lớp phân loại tuyến tính cuối cùng (`Fully Connected Layer`), chỉ giữ lại kiến trúc Backbone để trích xuất không gian đặc trưng phẳng mang **2.048 chiều** ($D=2048$).
- **Xử lý ngoại lệ:** Trong lớp `Dataset`, phải bọc khối lệnh đọc ảnh bằng cấu trúc `try-except`. Nếu gặp ảnh lỗi hoặc cấu trúc tệp hư hỏng, tự động trả về một ma trận Tensor rỗng (`torch.zeros(3, 224, 224)`) để pipeline không bị ngắt quãng giữa chừng.

### CELL 5: Pipeline Trích xuất Đặc trưng Ngữ nghĩa Văn bản (CLIP Text)

Xây dựng luồng xử lý ngôn ngữ tự nhiên để khai thác sâu thông tin ngữ nghĩa ẩn chứa trong trường tiêu đề sản phẩm (`title`).

- **Cơ chế tối ưu:** Tương tự Cell 4, tự động kiểm tra và tái sử dụng tệp đặc trưng `clip_text_features.npy` nếu đã tồn tại.
- **Hàm làm sạch Text văn bản thô (Text Preprocessing):** Xây dựng một hàm nội bộ thực hiện tuần tự các bước: chuyển đổi toàn bộ chuỗi sang chữ thường (`lowercasing`), sử dụng biểu thức chính quy (`regex`) để lọc bỏ toàn bộ ký tự đặc biệt, dấu câu, các biểu tượng cảm xúc (emojis) và loại bỏ khoảng trắng thừa.
- **Mã hóa chuỗi (Tokenization):** Chia văn bản thành các batch có quy mô 64 dòng để đưa vào bộ mã hóa chuyên dụng của mô hình CLIP (ViT-B/32). Thiết lập tham số cắt chuỗi/đệm chuỗi nghiêm ngặt tại giới hạn 77 tokens (`truncate=True`) theo chuẩn phân phối của OpenAI để tạo ra không gian vector ngữ nghĩa mang **512 chiều** ($D=512$).

### CELL 6: Giải thuật Late Fusion kết hợp Không gian và Tích hợp Chỉ mục FAISS

Thực hiện cấu trúc lõi của phương pháp đề xuất nhằm tạo lập siêu không gian vector đa phương thức và tối ưu hóa chi phí thời gian tìm kiếm tuyến tính.

- **Thuật toán Late Fusion (Ghép nối đặc trưng muộn):**
  1. Áp dụng `torch.nn.functional.normalize()` (chuẩn hóa L2) lên ma trận `resnet_feats` (2048 chiều).
  2. Áp dụng `torch.nn.functional.normalize()` lên ma trận `clip_text_feats` (512 chiều).
  3. Sử dụng lệnh `torch.cat(..., dim=1)` để nối ngang hai ma trận đã chuẩn hóa thành một Siêu vector Đa phương thức duy nhất mang kích thước **2.560 chiều** ($D_{	ext{fusion}} = 2048 + 512 = 2560$).
  4. Áp dụng chuẩn hóa L2 một lần nữa trên Siêu vector tổng hợp này.
- **Tích hợp tăng tốc với FAISS:** Chuyển đổi ma trận kết quả sang định dạng dữ liệu NumPy `float32`. Khởi tạo cấu trúc chỉ mục **`faiss.IndexFlatIP`** (Flat Inner Product - phép tính toán tích vô hướng trên vector đã chuẩn hóa L2 tương đương chính xác với khoảng cách Cosine Similarity). Chuyển cấu trúc chỉ mục này vào GPU thông qua `faiss.index_cpu_to_gpu` để tối ưu hóa năng lực tính toán song song, sau đó nạp toàn bộ 34.250 siêu vector vào hệ thống chỉ mục. Lưu tệp ma trận phẳng thành tệp `fusion_features.npy`.

### CELL 7: Đánh giá Toàn diện Chỉ số Hiệu năng Phương pháp chính

Triển khai giao thức kiểm định thực nghiệm nghiêm ngặt trên toàn bộ quy mô Dataset để đo lường năng lực truy xuất sản phẩm.

- **Giao thức thực nghiệm:** Sử dụng vòng lặp duyệt qua từng dòng dữ liệu làm Query. Sử dụng hàm `index.search(query, MAX_K + 1)` của FAISS để tìm kiếm $K+1$ phần tử lân cận gần nhất. Viết logic loại bỏ chính chỉ số index của Query khỏi danh sách kết quả trả về (`Self-Similarity Removal`), sau đó cắt lấy danh sách Top-K ($K=5$) đáp án cuối cùng.
- **Độ đo toán học yêu cầu:** Tính toán chi tiết các chỉ số tại các ngưỡng chặn kỹ thuật $K \in [1, 3, 5, 10]$:
  - `Precision@K`: Số lượng sản phẩm trả về đúng nhóm chia cho $K$.
  - `Recall@K`: Số lượng sản phẩm đúng tìm thấy chia cho tổng số lượng sản phẩm thực tế của nhóm nhãn đó trong Dataset (ngoại trừ chính nó).
  - `mAP@5`: Diện tích dưới đường cong Precision-Recall dựa trên thứ tự xếp hạng (ranking) của các phần tử chính xác trong không gian Top-5.
- **Xuất dữ liệu:** Kết xuất bảng điểm trung bình tổng quan ra tệp `metrics_fusion.csv` và bảng điểm AP chi tiết của từng Query đơn lẻ ra tệp `fusion_detail_metrics.csv` trong thư mục `RESULTS`.

### CELL 8: Trực quan hóa Phân tích lỗi và Minh chứng Thực nghiệm

Xây dựng hệ thống hàm đồ họa trực quan sinh động bằng `matplotlib` phục vụ việc viết báo cáo minh chứng cá nhân (Mục 7 và Mục 8 trong file Word).

- **Cơ chế tìm kiếm tự động:** Viết đoạn code thuật toán tự động lọc ra từ tập kết quả:
  - Chọn ra 2 trường hợp **Mẫu đúng hoàn toàn (True Positives)**: Vị trí Top-1 trả về kết quả chính xác tuyệt đối (trùng `label_group`).
  - Chọn ra 2 trường hợp **Mẫu sai sót (False Positives)**: Vị trí Top-1 trả về kết quả sai hoàn toàn (khác `label_group` của Query).
- **Quy chuẩn hiển thị hình ảnh:** Với mỗi mẫu tìm được, vẽ một hàng ngang gồm 6 phân nhánh ảnh: Ảnh đầu tiên ở góc lề trái là ảnh `QUERY` gốc; 5 ảnh tiếp theo là danh sách kết quả `Top-1` đến `Top-5` được hệ thống xếp hạng từ trái sang phải.
- **Trang trí đồ họa học thuật:** Ảnh đúng phải hiển thị khung tiêu đề chữ viết màu **Xanh lá cây** kèm dấu tích ($\checkmark$). Ảnh sai phải hiển thị khung tiêu đề màu **Đỏ** kèm dấu gạch chéo ($ imes$). Tiêu đề của mỗi ảnh kết quả phải in rõ chỉ số xếp hạng, điểm số khoảng cách tương đồng không gian (Distance) và chuỗi văn bản tiêu đề sản phẩm (`title`) đã được cắt ngắn gọn. Tự động lưu toàn bộ các biểu đồ này thành các tệp ảnh `.png` tương ứng vào thư mục `RESULTS`.

### CELL 9: Khung Nhận xét, Biên luận Thực nghiệm Học thuật

Sinh ra một cell định dạng văn bản Markdown bằng tiếng Việt, ghi nhận sẵn cấu trúc 6 luận điểm nhận xét chuyên sâu theo barem chấm điểm của Giảng viên:

1.  **Biện luận chỉ số mAP@5:** Đánh giá mức độ tăng trưởng vượt trội của chỉ số mAP phương pháp chính (Multimodal Late Fusion) so với các baseline đơn lẻ đơn mục tiêu tuần trước (ResNet50, pHash, TF-IDF). Giải thích cơ chế tương hỗ hình học.
2.  **Ý nghĩa thực tiễn của Precision và Recall:** Phân tích ý nghĩa của tỷ lệ Precision@1 đối với trải nghiệm mua sắm của người dùng trên môi trường thương mại điện tử thực tế.
3.  **Ưu điểm bản chất hệ thống Đa phương thức:** Cơ chế không gian ngữ nghĩa văn bản của CLIP Text giải cứu các lỗi sai số do nhiễu thị giác cục bộ (yếu tố phông nền lộn xộn, góc chụp lệch, điều kiện ánh sáng kém) của mạng tích chập ResNet50.
4.  **Mổ xẻ nguyên nhân gây lỗi False Positives (Dựa trên Cell 8):** Phân tích các trường hợp mô hình bị đánh lừa do người bán cố tình đặt tiêu đề spam từ khóa sai lệch (Textual Noise), hoặc các sản phẩm có bao bì giống hệt nhau nhưng chỉ khác biệt chi tiết cực nhỏ (Fine-grained classification).
5.  **Luận điểm cốt lõi về FAISS:** Khẳng định rõ ràng luận điểm giảng viên yêu cầu: Tích hợp thư viện FAISS IndexFlatIP bản chất toán học chỉ tương đương toán tử tính toán khoảng cách Cosine truyền thống, không tự làm thay đổi hay tăng mAP. Tuy nhiên, cấu trúc này tối ưu chi phí thời gian từ mức tuyến tính $O(N)$ xuống mức mili-giây, giúp bài toán khả thi trên quy mô lớn.
6.  **Hạn chế và Hướng đi Tuần 4:** Đánh giá nhược điểm của mô hình khi vận hành ở trạng thái Zero-shot (trọng số gốc chưa được tinh chỉnh), đặt nền móng cho kỹ thuật Học chỉ số khoảng cách (Metric Learning - ArcFace Loss) ở Tuần sau.

---

## III. CHỈ THỊ THỰC THI CHO AI ASSISTANT (INSTRUCTION FOR VIBECODING)

- Bạn hãy đọc kỹ và tuân thủ tuyệt đối toàn bộ cấu trúc logic, giải thuật và quy tắc đặt tên biến, tên tệp kết xuất đã được quy định trong tài liệu hướng dẫn này.
- Hãy sinh mã nguồn hoàn chỉnh, không viết mã nguồn dạng tóm tắt hoặc chèn chú thích dạng `# TODO: viết tiếp code ở đây`. Toàn bộ code trong các cell phải chạy được ngay lập tức trên môi trường Google Colab mà không sinh ra bất kỳ dòng thông báo lỗi đỏ nào.
- **Lệnh thực thi cài đặt:** Chèn dòng mã lệnh `!pip install faiss-gpu git+https://github.com/openai/CLIP.git` ngay đầu cell 1 hoặc cấu hình chú thích hướng dẫn chạy cụ thể để bảo đảm môi trường không bị thiếu thư viện nguồn.
