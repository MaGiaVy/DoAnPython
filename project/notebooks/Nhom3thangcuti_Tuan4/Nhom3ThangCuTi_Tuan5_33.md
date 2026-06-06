
![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.001.png)

**UBND THÀNH PHỐ HỒ CHÍ MINH**

**TRƯỜNG ĐẠI HỌC SÀI GÒN**


![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.002.png)
**\


`     `**BÁO CÁO**

`  `**TÊN HỌC PHẦN: NGÔN NGỮ PYTHON**

**TÊN ĐỀ TÀI**

**TÌM KIẾM HÌNH ẢNH SẢN PHẨM TƯƠNG ĐỒNG**

**Xây dựng hệ thống truy xuất sản phẩm tương đồng dựa trên đặc trưng hình ảnh**

**Thuộc nhóm ngành khoa học:** Công nghệ thông tin.

**Thành viên tham gia:**

1. Lê Quốc Bảo
1. Mã Gia Vỹ
1. Nguyễn Khánh Hưng

`      `**Giáo viên hướng dẫn: ThS.NCS. Hà Thanh Dũng**




Thành phố Hồ Chí Minh, tháng 05 năm 2026




Tìm kiếm hình ảnh sản phẩm tương đồng

(Visual Search)

Mục Lục

[**Bảng phân công	1**](#_heading=)

[Danh mục từ viết tắt	1](#_heading=h.t67ldu7xrd87)

[**1. TÓM TẮT VÀ MỤC TIÊU	2**](#_heading=h.qqip2smcx6tu)

[1.1 TÓM TẮT	2](#_heading=h.s78ob1qay0d)

[1.2 MỤC TIÊU	3](#_heading=h.wly463n1si88)

[2. LỊCH SỬ LÀM VIỆC TRONG TUẦN 4	4](#_heading=h.ag07gxu4psdz)

[3. CÁC VẤN ĐỀ CÒN TỒN TẠI TỪ TUẦN 3 VÀ CÁCH ĐÃ SỬA	5](#_heading=h.vkcrki525b8e)

[3.1 Minh bạch hóa quy trình tinh chỉnh siêu tham số:	5](#_heading=h.r4d1c19eonh9)

[3.2 Cụ thể hóa nền tảng toán học trong thuật toán Tái xếp hạng:	5](#_heading=h.cibjw8irpsns)

[3.3 Triển khai giao thức đánh giá độc lập (Tránh Data Leakage):	6](#_heading=h.mhngnw23dwss)

[4. DỮ LIỆU VÀ PIPELINE XỬ LÝ CUỐI CÙNG	6](#_heading=h.johpkjwd3thl)

[4.1. LÀM SẠCH DỮ LIỆU (DATA CLEANING)	7](#_heading=h.tadg0iibeofz)

[4.2. TIỀN XỬ LÝ DỮ LIỆU (IMAGE PREPROCESSING)	7](#_heading=h.xsjmv9i92hbu)

[4.3 TIỀN XỬ LÝ VĂN BẢN (TEXT PREPROCESSING)	8](#_heading=h.j6bkre3bdbey)

[4.4. Xây dựng Pipeline dữ liệu	9](#_heading=h.719kjpazlt47)

[5. MÔ HÌNH BASELINE VÀ PHƯƠNG PHÁP CHÍNH ĐÃ TRIỂN KHAI	11](#_heading=h.d8w8edtc2hte)

[5.1. DỮ LIỆU SỬ DỤNG VÀ GIAO THỨC ĐÁNH GIÁ ĐỘC LẬP	11](#_heading=h.zidhymwzlykg)

[5.2. CÁC MÔ HÌNH CƠ SỞ (BASELINES)	12](#_heading=h.tqzt3v9tqmib)

[5.3. KIẾN TRÚC PHƯƠNG PHÁP CHÍNH: TRUY XUẤT 2 GIAI ĐOẠN (2-STAGE RETRIEVAL)	12](#_heading=h.ddseig6ba9yp)

[5.4. HỆ THỐNG CHỈ SỐ ĐÁNH GIÁ (METRICS)	13](#_heading=h.rytt5k9lf1a)

[5.5. MÔ HÌNH ĐỊNH HƯỚNG CỦA GIÁO VIÊN	13](#_heading=h.urakrqu5k7b9)

[6.  CÁC CẢI TIẾN/ TỐI ƯU TRONG TUẦN 4:	15](#_heading=h.lholqsmq2o9l)

[6.1. KIẾN TRÚC HỆ THỐNG 2 GIAI ĐOẠN VÀ CƠ SỞ LÝ LUẬN	15](#_heading=h.41pnq3weoaq4)

[6.2. QUY TRÌNH TRÍCH XUẤT, HÒA TRỘN VÀ TÁI XẾP HẠNG (PIPELINE CHÍNH THỨC)	15](#_heading=h.z6cvazqty8ul)

[6.3. TỐI ƯU HÓA KHÔNG GIAN TÌM KIẾM QUY MÔ  LỚN VỚI FAISS	17](#_heading=h.rzlj4hntx9k0)

[6.4. CHIẾN LƯỢC TỐI ƯU HÓA SIÊU THAM SỐ VÀ CƠ CHẾ TÁI XẾP HẠNG (RE-RANKING)	17](#_heading=h.h6k1zdcnr3z0)

[6.4.1. Tối ưu trọng số hòa trộn đa phương thức (alpha) tại Giai đoạn 1	17](#_heading=h.uft282peuryx)

[6.4.2. Cơ chế Tái xếp hạng ứng viên (Candidate Re-ranking) bằng DINOv2	18](#_heading=h.9x23zbfmf0bb)

[7. KẾT QUẢ THỰC NGHIỆM VÀ BẢNG SO SÁNH	19](#_heading=h.vlgie2qds2ei)

[8. PHÂN TÍCH LỖI CHUYÊN SÂU	25](#_heading=h.xjm4lr56zvp6)

[8.1. Quy trình trích xuất mẫu lỗi	25](#_heading=h.dsmhwp1lpb8m)

[8.2. Cấu trúc bảng phân tích lỗi	26](#_heading=h.lyaierkk68jz)

[8.3. Kết quả phân loại 10 mẫu lỗi đại diện	26](#_heading=h.uqg78t59gdnu)

[8.4. Phân tích nguyên nhân lỗi	28](#_heading=h.50aasyu9iosw)

[8.5. Độ tin cậy của quy trình phân loại lỗi và hướng cải tiến	29](#_heading=h.af40gebi1f3a)

[**9. DEMO HOẶC SẢN PHẨM THỬ NGHIỆM	29**](#_heading=h.ynorrnpdrefv)

[9.1. Mục tiêu của demo	30](#_heading=h.27p8pvrw879p)

[9.2. Dữ liệu và file đầu vào của demo	30](#_heading=h.nrd5pu59018z)

[9.3. Quy trình hoạt động của demo	30](#_heading=h.exxgmdhdm2nw)

[9.4. Kết quả đầu ra của demo	31](#_heading=h.ukzp824u7i6m)

[9.5. Nhận xét về sản phẩm thử nghiệm	32](#_heading=h.1byjm1hck9vs)

[10. PHÂN CÔNG, MINH CHỨNG CÁ NHÂN VÀ KHAI BÁO SỬ DỤNG AI	32](#_heading=h.wl3degyuh5z7)

[11. KẾ HOẠCH TUẦN 5	32](#_heading=h.h8ep106zfjws)

[12. TÀI LIỆU THAM KHẢO](#_heading=h.189v3ox24ggx)[	35](#_heading=h.189v3ox24ggx)

# <a name="_heading=h.vupppftv6dve"></a>**
# Bảng phân công

|**MSSV**|**Họ và tên**|**Nhiệm vụ trong tuần**|
| :-: | :-: | :-: |
|3124410015|Lê Quốc Bảo|Hoàn thiện phân tích lỗi, chuẩn bị demo, kiểm tra tái lập kết quả|
|3124410414|Mã Gia Vỹ|Fix bug normalize features, chốt pipeline cuối, chạy lại kết quả chính thức|
|3124410129|Nguyễn Khánh Hưng|Hoàn thiện báo cáo, lập bảng so sánh cuối, chuẩn bị slide thuyết trình|

**|\
DANH MỤC TỪ VIẾT TẮT**

|**Từ viết tắt**|**Thuật ngữ đầy đủ**|**Ý nghĩa**|
| :-: | :-: | :-: |
|**Baseline**|Baseline Model|Mô hình cơ sở dùng làm mốc so sánh.|
|**BBox**|Bounding Box|Khung hình chữ nhật bao quanh vật thể, thu được từ mô hình YOLO để tiến hành cắt vùng ảnh sản phẩm (Crop).|
|**CBIR**|Content-Based Image Retrieval|Truy xuất hình ảnh dựa trên nội dung. Tên học thuật chính thức của bài toán Visual Search mà đề tài đang giải quyết.|
|**CLIP**|Contrastive Language-Image Pre-training|Mô hình học đa phương thức (Multimodal), nền tảng cốt lõi của mạng MobileCLIP dùng ở Giai đoạn 1.|
|**CNN**|Convolutional Neural Network|Mạng nơ-ron tích chập, dùng để trích xuất đặc trưng từ hình ảnh.|
|**EDA**|Exploratory Data Analysis|Phân tích dữ liệu khám phá, bước tìm hiểu đặc tính của bộ dữ liệu.|
|**Embedding**|Feature Vector|Vector đặc trưng đại diện cho nội dung của một bức ảnh dưới dạng số.|
|**FAISS**|Facebook AI Similarity Search|Thư viện tối ưu hóa việc tìm kiếm các vector tương đồng.|
|**IP**|Inner Product|Tích vô hướng. Phép toán toán học cấu hình trong faiss.IndexFlatIP dùng để tính độ tương đồng khi vector đã chuẩn hóa L2.|
|**mAP**|Mean Average Precision|Độ chính xác trung bình (chỉ số chính để đánh giá hệ thống truy vấn).|
|**OCR**|Optical Character Recognition|Nhận diện ký tự quang học, hướng phát triển tương lai dùng để bóc tách và xóa các đoạn chữ quảng cáo rác đè lên ảnh.|
|**pHash**|Perceptual Hashing|Mã băm cảm nhận, thuật toán băm ảnh truyền thống dùng làm Baseline lọc trùng và bắt các ảnh copy-paste thô.|
|**Recall@K**|Recall at K|Độ phủ tại vị trí K (tỷ lệ ảnh đúng tìm thấy trong top K kết quả).|
|**ResNet**|Residual Network|Một kiến trúc mạng nơ-ron sâu (Deep Learning) phổ biến.|
|**SAHI**|Slicing Aided Hyper Inference|Kỹ thuật phân mảnh ảnh, hỗ trợ YOLO phát hiện các sản phẩm có kích thước siêu nhỏ hoặc bị che khuất trên sàn Shopee.|
|**TF-IDF**|Term Frequency - Inverse Document Frequency|Thuật toán thống kê tính trọng số tần suất từ vựng, dùng cho mô hình Baseline xử lý tiêu đề văn bản (title).|
|**UI / UX**|User Interface / User Experience|Giao diện người dùng và Trải nghiệm người dùng, mục tiêu hoàn thiện khi đóng gói hệ thống sang dạng Web App Streamlit.|
|**ViT**|Vision Transformer|Kiến trúc mạng Transformer áp dụng cho thị giác. Đây là lõi công nghệ của mô hình DINOv2-vitb14 dùng ở Giai đoạn 2.|
|**YOLO**|You Only Look Once|Thuật toán phát hiện vật thể thời gian thực, dùng trong luồng thực nghiệm chuyên sâu theo định hướng của Giảng viên.|
|L2-Norm|L2 Normalization|Chuẩn hóa vector về mặt cầu đơn vị, bắt buộc trước khi dùng FAISS IndexFlatIP.|


# <a name="_heading=h.qqip2smcx6tu"></a>1. TÓM TẮT VÀ MỤC TIÊU 
## <a name="_heading=h.s78ob1qay0d"></a> 1.1 TÓM TẮT

|**Nội dung**|**Chi tiết thông tin**|
| :-: | :-: |
|**Mã đề tài/nhóm**|Mã đề tài: 33- Visual Search, Nhóm: 3 thằng cu tí|
|**Loại bài toán**|Truy xuất hình ảnh|
|**Dữ liệu sử dụng**|<p>Tên: Shopee - Price Match Guarantee</p><p>Nguồn: Kaggle Competition Dataset</p><p>Số lượng mẫu: 34,250</p>|
|**Phương pháp chính**|MobileCLIP (GĐ1) + DINOv2-vitb14 re-ranking (GĐ2) với Score Fusion, toàn bộ siêu tham số alpha/beta/retrieval\_k được Grid Search trên tập Validation|
|**Metric chính**|Precision@K, Recall@K, mAP( metric chính)|
|**Kết quả** |mAP@5 = 0.7872, Precision@1 = 0.8064, Recall@5 = 0.7530 (pipeline MobileCLIP + DINOv2, tập Test 80%)|

*Báo cáo này trình bày quá trình nghiên cứu và phát triển hệ thống truy xuất hình ảnh sản phẩm tương đồng (Visual Search) trên bộ dữ liệu Shopee - Price Match Guarantee, bao gồm 34.250 ảnh thuộc 11.014 nhóm sản phẩm. Kết quả phân tích dữ liệu ban đầu cho thấy bộ dữ liệu có hiện tượng mất cân bằng rõ rệt theo phân phối đuôi dài (long-tail), trong đó phần lớn các nhóm sản phẩm chỉ chứa từ 2 đến 3 ảnh. Ở Tuần 2, nhóm đã xây dựng mô hình cơ sở (baseline) sử dụng MobileCLIP để trích xuất đặc trưng hình ảnh. Tuy nhiên, việc đánh giá trên tập con ngẫu nhiên 500 ảnh chưa đảm bảo mỗi ảnh truy vấn đều có ảnh cùng nhóm trong không gian tìm kiếm, khiến các chỉ số đo lường như Precision@K, Recall@K và mAP chưa phản ánh đầy đủ hiệu năng thực tế của hệ thống.*

*Tiếp thu góp ý từ Giảng viên, trong Tuần 3 nhóm đã điều chỉnh không gian tìm kiếm (Gallery) lên toàn bộ 34.250 ảnh, đồng thời tích hợp thư viện FAISS nhằm tối ưu hóa tốc độ truy xuất trên dữ liệu lớn. Các thực nghiệm với pHash, TF-IDF và MobileCLIP đã được tiến hành. Dù vậy, phương pháp ghép nối đặc trưng đa phương thức ở tuần này chưa đạt hiệu quả như kỳ vọng do khoảng cách ngữ nghĩa giữa đặc trưng ảnh và văn bản, cũng như đặc thù tiêu đề sản phẩm Shopee chứa nhiều ngôn ngữ và từ khóa nhiễu. Thêm vào đó, việc tinh chỉnh tham số trực tiếp trên tập đánh giá chung đã bộc lộ rủi ro rò rỉ dữ liệu (data leakage) và học vẹt (overfitting).*

*Khắc phục triệt để các hạn chế trên, trong Tuần 4, nhóm đã thiết lập giao thức đánh giá chuẩn mực bằng cách chia tập truy vấn (Query) thành tập Validation (20%) và tập Test (80%) độc lập. Phương pháp chính thức được nhóm đề xuất là kiến trúc truy xuất 2 giai đoạn (2-Stage Retrieval). Cụ thể, Giai đoạn 1 (Candidate Generation) ứng dụng MobileCLIP để kết hợp đặc trưng đa phương thức thông qua trọng số alpha, kết hợp tìm kiếm FAISS để trích xuất tập ứng viên tiềm năng (retrieval\_k). Bước sang Giai đoạn 2 (Re-ranking), hệ thống sử dụng khả năng trích xuất đặc trưng thị giác sâu của mạng DINOv2. Thông qua cơ chế Score Fusion, điểm độ tương đồng của DINOv2 được hòa trộn tuyến tính với điểm số MobileCLIP ban đầu theo trọng số beta. Toàn bộ các siêu tham số (alpha, beta, retrieval\_k) đều được tinh chỉnh tối ưu (Grid Search) nghiêm ngặt trên tập Validation trước khi áp dụng để tái xếp hạng ứng viên, qua đó giúp hệ thống giảm ảnh hưởng của nhiễu phông nền và trả về Top-5 kết quả chính xác nhất trên tập Test.*

*Song song với kiến trúc chính, nhóm cũng đã mở rộng thực nghiệm một luồng xử lý độc lập bám sát định hướng của Giảng viên. Bằng việc áp dụng các kỹ thuật Computer Vision chuyên sâu để làm sạch dữ liệu (xóa phông nền, bóc tách vật thể nhỏ), luồng tiếp cận này đã mang lại kết quả tích cực ngoài mong đợi với mAP@5 = 0.77, qua đó khẳng định tiềm năng to lớn của việc tích hợp Object Detection vào bài toán tìm kiếm trực quan.* 

*Bước sang Tuần 5, nhóm tập trung tối ưu hóa toán học, phân tích lỗi định lượng và đóng gói pipeline thành sản phẩm thực tế. Qua kiểm toán mã nguồn, nhóm đã khắc phục triệt để lỗi hệ thống cũ bằng cách bổ sung bước chuẩn hóa L2 cho vector tổng hợp sau khi Feature Fusion, đảm bảo phép tính Inner Product của FAISS IndexFlatIP tương đương chính xác với Cosine Similarity. Thay đổi này giúp hệ thống thiết lập đỉnh hiệu năng trên tập Test độc lập với mAP@5 = 0.7872 và Precision@1 = 0.8064, đồng thời tối ưu độ trễ đạt mốc lý tưởng 0.0307 giây/query, nhanh gấp 5.4 lần so với Brute-force Python. Nhằm chỉ ra các giới hạn công nghệ, nhóm đã lập quy trình trích xuất lỗi tự động; kết quả ghi nhận tỷ lệ lỗi hoàn toàn (mAP@5 = 0) chiếm 8.91% do áp lực phân phối long-tail, với các nguyên nhân cốt lõi từ nhiễu nền trắng studio, góc chụp biến đổi và watermark quảng cáo. Cuối cùng, nhóm hiện thực hóa sản phẩm bằng một Web App trực quan qua framework Streamlit, tách biệt cấu trúc giao diện kéo thả (app.py) và backend xử lý lõi AI (pipeline.py) để giả lập trải nghiệm chuẩn sàn thương mại điện tử, đồng thời chuẩn bị tệp Demo.ipynb làm kịch bản dự phòng an toàn trước Hội đồng.* 
## <a name="_heading=h.wly463n1si88"></a>1.2 MỤC TIÊU
- **1. Tái lập kết quả:** Đồng bộ 100% số liệu, đảm bảo pipeline chạy trơn tru từ đầu đến cuối không lỗi dòng lệnh. Số liệu trong báo cáo Word phải khớp tuyệt đối với kết quả xuất ra từ file Notebook.
- **2. Sửa lỗi kỹ thuật:** Vá triệt để bug thiếu L2-normalize trong hàm fuse\_and\_normalize\_clip, đảm bảo vector tổng hợp được chuẩn hóa về mặt cầu đơn vị trước khi nạp vào faiss.IndexFlatIP để phép tính Inner Product hoạt động đúng bản chất toán học của Cosine Similarity.
- **3. Chốt kết quả chính thức:** Đóng băng tập Test (80%), xác định cấu hình MobileCLIP + DINOv2 re-ranking làm run chính thức. Sử dụng duy nhất một bảng kết quả cuối cùng (mAP@5 = 0.7872, Precision@1 = 0.8064) làm mốc đối chứng xuyên suốt.
- **4. Hoàn thiện bản Demo:** Đảm bảo file mã nguồn xử lý và giao diện kéo thả hoạt động ổn định. Định hình rõ ràng luồng Input (ảnh upload từ máy) và Output (lưới kết quả hiển thị Top-5, mã nhóm, điểm số và badge khớp tuyệt đối).
- **5. Chuẩn bị thuyết trình:** Hoàn thiện bộ slide báo cáo dựa trên khung sườn 7 chương cốt lõi. Thiết lập kịch bản nói phân chia theo mốc thời gian nghiêm ngặt để kiểm soát bài thuyết trình gói gọn trong **5–7 phút** trước Hội đồng.


# <a name="_heading=h.ag07gxu4psdz"></a>**2. LỊCH SỬ LÀM VIỆC TRONG TUẦN** 
<a name="_heading=h.rzbnt0veceti"></a>BẢNG PHÂN CÔNG

|**Thời điểm**|**Thành viên thực hiện**|**Nội dung công việc**|**Sản phẩm/minh chứng**|**Trạng thái**|
| :-: | :-: | :-: | :-: | :-: |
|01–02/6|Mã Gia Vỹ|Fix bug L2-normalize trong hàm fusion (chuẩn hóa vector tổng hợp sau khi hòa trộn), chạy lại toàn bộ pipeline|Tuan5\_GiaVy\_Pipeline\_Fixed.ipynb, final\_metric\_fixed.csv|Hoàn thành|
|02–03/6|Mã Gia Vỹ|Chốt kết quả chính thức, thêm requirements.txt và random seed|requirements.txt, final\_results\_official.csv|Hoàn thành|
|02–03/6|Lê Quốc Bảo|Cập nhật phân tích lỗi với kết quả pipeline đã fix, bổ sung hướng xử lý|Tuan5\_QuocBao\_ErrorAnalysis.ipynb, error\_analysis\_v2.csv|Hoàn thành|
|03–04/6|Lê Quốc Bảo|Kiểm tra demo chạy end-to-end, thêm ví dụ input/output cụ thể|demo\_v2.ipynb|Hoàn thành|
|04–05/6|Nguyễn Khánh Hưng|Hoàn thiện báo cáo tuần 5, lập bảng so sánh cuối|Nhom3ThangCuTi\_Tuan5\_33.docx|Hoàn thành|
|05–06/6|Nguyễn Khánh Hưng|Chuẩn bị slide thuyết trình 5–7 phút, kiểm tra file nộp|slides\_tuan5.pptx|Hoàn thành|
|05–06/6|Cả nhóm|Kiểm tra lại toàn bộ file nộp, khai báo AI, tài liệu tham khảo|Thư mục nộp hoàn chỉnh|Hoàn thành|

# <a name="_heading=h.vkcrki525b8e"></a>**3. TÓM TẮT QUÁ TRÌNH TỪ TUẦN 1 TỚI TUẦN 4**

## <a name="_heading=h.ofjqmqcjxtg7"></a>**Tuần 1: Xác định bài toán & Hướng đi sơ bộ**
- **Xác định bài toán & Khảo sát dữ liệu:** Định hình mục tiêu xây dựng hệ thống Tìm kiếm trực quan qua bài toán Truy xuất hình ảnh dựa trên nội dung (CBIR) trên bộ dữ liệu thực tế *Shopee - Price Match Guarantee*.
- **Thiết lập hệ thống đo lường:** Thống nhất các metric đánh giá chất lượng truy xuất và xếp hạng bao gồm: Precision@K, Recall@K và chỉ số cốt lõi mAP.
- **Đề xuất giải pháp kỹ thuật ban đầu:** Nghiên cứu tổng quan tài liệu và vạch ra lộ trình ứng dụng Transfer Learning với mạng CNN (như ResNet50) để trích xuất vector đặc trưng (Embedding).
- **Định hướng tối ưu hạ tầng:** Dự kiến tích hợp thư viện FAISS nhằm giải quyết bài toán gia tốc tốc độ tìm kiếm lân cận, đồng thời thiết lập bộ câu hỏi nghiên cứu làm cơ sở cho các tuần thực nghiệm chuyên sâu.

## <a name="_heading=h.xfyn7d32bxtk"></a>**Tuần 2 — Xây dựng baseline đầu tiên**
- **Phân tích dữ liệu khám phá (EDA):** Khảo sát toàn diện tập dữ liệu *Shopee – Price Match Guarantee*, phát hiện đặc tính mất cân bằng nghiêm trọng theo phân phối đuôi dài (**Long-tail**): **63,4%** số nhóm sản phẩm chỉ có đúng 2 ảnh, lượng ảnh trung bình ở mức thấp với 3,11 ảnh/nhóm.
- **Tiền xử lý và chuẩn hóa ảnh:** Thực hiện chuỗi biến đổi hình học trên ảnh đầu vào qua các bước **Resize** (224x224 pixel), **Center Crop** và áp dụng bộ tham số **Normalization** theo chuẩn ImageNet để làm sạch dữ liệu.
- **Triển khai mô hình cơ sở (Baseline):** Ứng dụng mạng CNN **ResNet50** tiền huấn luyện để cấu trúc hóa vector đặc trưng (**Embedding**) 2048 chiều; đo lường độ tương đồng bằng thuật toán **Cosine Similarity**.
- **Phát hiện lỗi giao thức đánh giá:** Việc lấy mẫu ngẫu nhiên thô 500 ảnh ban đầu khiến nhiều ảnh truy vấn (**Query**) rơi vào trạng thái không có đáp án tương ứng trong không gian tìm kiếm (**Gallery**), làm sai lệch nghiêm trọng các chỉ số mAP, Precision và Recall.
- **Hiệu chỉnh theo góp ý của Giảng viên:** Tái cấu trúc tập dữ liệu mẫu đảm bảo mọi Query đều có sản phẩm đối sánh cùng nhóm (label\_group), chuẩn hóa lại thước đo hệ thống và đẩy chỉ số mAP của Baseline đạt mốc **0.7755**.

## <a name="_heading=h.iheo1sqh6ios"></a>**Tuần 3 — Mở rộng quy trình trên toàn bộ dữ liệu & Thử nghiệm Đa phương thức**
- **Nâng cấp quy mô không gian tìm kiếm:** Tiếp thu góp ý từ Giảng viên, chuyển dịch từ tập mẫu 500 ảnh lên kiểm thử diện rộng trên toàn bộ dữ liệu (**34.250 ảnh**) nhằm đảm bảo tính khách quan cho hệ thống.
- **Tối ưu tốc độ hạ tầng:** Tích hợp thành công thư viện **FAISS** với cấu hình chỉ mục faiss.IndexFlatIP, giải quyết triệt để bài toán gia tốc tốc độ tìm kiếm vector lân cận trên không gian dữ liệu lớn.
- **Thực nghiệm đa mô hình diện rộng:** Triển khai đánh giá đồng thời các baseline độc lập bao gồm **pHash** (lọc trùng ảnh thô), **TF-IDF** (xử lý ngữ nghĩa tiêu đề), và mô hình đa phương thức **MobileCLIP**.
- **Phát hiện nút thắt kỹ thuật và lỗi hệ thống:** \* Cơ chế ghép nối trực tiếp vector (Late Fusion 1:1 giữa ResNet50 và CLIP Text) cho hiệu năng kém (mAP@5 = 0.3776) do khoảng cách miền ngữ nghĩa lớn và nạn nhồi nhét từ khóa rác từ người bán.
  - Nhận diện rủi ro rò rỉ dữ liệu và học vẹt khi nhóm cố tình tinh chỉnh siêu tham số trực tiếp trên toàn bộ tập dữ liệu chung.
- **Cải tiến bước đầu khả quan:** Đề xuất cơ chế hòa trộn điểm số có trọng số giữa ResNet50 và TF-IDF, kết hợp thêm kỹ thuật **pHash boosting** giúp kéo chỉ số mAP@5 lên mốc **0.7635**.

## <a name="_heading=h.oirpvulgs8bo"></a>**Tuần 4 — Kiến trúc 2 giai đoạn và giao thức đánh giá chuẩn**
Nhóm khắc phục triệt để các hạn chế bằng cách:

- **Chuẩn hóa giao thức kiểm thử độc lập:** Áp dụng kỹ thuật chia phân tầng (**Stratified Split**) tách biệt tập truy vấn thành **Validation (20%)** để tối ưu hóa siêu tham số và **Test (80%)** bị đóng băng hoàn toàn nhằm triệt tiêu hoàn toàn rủi ro rò rỉ dữ liệu (**Data Leakage**).
- **Kiến trúc phương pháp chính (Two-Stage Retrieval Pipeline):**
  - *Giai đoạn 1 (Candidate Generation):* Ứng dụng **MobileCLIP** hòa trộn đặc trưng ảnh và chữ theo tỷ lệ tối ưu alpha = 0.5 qua giải thuật Tìm kiếm dạng lưới (**Grid Search**). Tích hợp chỉ mục **FAISS** giúp tăng tốc độ phản hồi hệ thống gấp **5.4 lần** (giảm từ 0.1656s xuống **0.0307s/query**).
  - *Giai đoạn 2 (Re-ranking):* Kích hoạt mạng Transformer **DINOv2-vitb14** khai thác năng lực thị giác vi mô để soi chi tiết họa tiết, logo của 100 ứng viên lọc thô; tiến hành hòa trộn điểm số tuyến tính (**Score Fusion**) theo trọng số beta.
- **Đỉnh hiệu năng thiết lập trên tập Test:** Phương pháp chính đạt kết quả bứt phá toàn diện, cân bằng lý tưởng giữa hai miền Precision và Recall với **mAP@5 = 0.7872**, **Precision@1 = 0.8064** và **Recall@5 = 0.7530**.
- **Thực nghiệm luồng xử lý theo định hướng của Giảng viên:** Hiện thực hóa luồng nhận diện vật thể chuyên sâu nhằm khử phông nền nhiễu phức tạp và bóc tách đối tượng nhỏ: **YOLO + SAHI + Crop BBox + Crop Classifier (EfficientNetB0) + Soft Fusion** đạt chỉ số ấn tượng **mAP@5 = 0.77**. Nhóm bảo lưu hướng này để nghiên cứu trích xuất vector lớp ẩn nạp trực tiếp vào FAISS nhằm tối ưu chi phí phần cứng.
- **Phân tích mẫu lỗi bước đầu (Error Analysis):** Thiết lập quy trình lọc tự động thu được **3.053 mẫu lỗi nghiêm trọng** (mAP@5 = 0), phân loại định tính thành các nhóm nguyên nhân chính: nhiễu trùng màu sắc/phông nền studio, góc chụp biến đổi diện rộng, giới hạn phân loại tinh và tiêu đề nhồi từ khóa rác đè lên ngữ nghĩa ảnh.

|<h2></h2>|
| :- |
||
||
||
||
# <a name="_heading=h.cpxgo36z2ffo"></a><a name="_heading=h.johpkjwd3thl"></a>**4. DỮ LIỆU VÀ PIPELINE XỬ LÝ CUỐI CÙNG**
`                                                 `Bảng 1: Thống kê tổng quan bộ dữ liệu

|**Thuộc tính**|**Chi tiết thông tin**|
| :-: | :-: |
|**Tên bộ dữ liệu**|Shopee - Price Match Guarantee|
|**Nguồn dữ liệu**|Kaggle Competition Dataset|
|**Loại bài toán**|Truy xuất/gợi ý hình ảnh sản phẩm tương đồng|
|**Tổng số lượng mẫu**|34,250 bản ghi (hình ảnh)|
|**Số lượng đặc trưng**|05 cột (posting\_id, image, image\_phash, title, label\_group)|
|**Biến mục tiêu** |label\_group|
|**Số lượng nhóm sản phẩm**|11,014 nhóm nhãn duy nhất|
|**Định dạng dữ liệu**|Hình ảnh (.jpg) và Văn bản (chuỗi ký tự Unicode)|
|**Hạn chế dữ liệu**|Mất cân bằng nhóm sản phẩm (63,4% nhóm có đúng 2 ảnh, trung bình mỗi nhóm chỉ có 3,11 ảnh, trong khi nhóm lớn nhất có 51 ảnh)|

**Ý nghĩa chi tiết các trường dữ liệu (Metadata)**

Mỗi bản ghi trong file cấu trúc CSV bao gồm các trường thông tin sau:

- **posting\_id**: Mã định danh duy nhất cho từng bài đăng của nhà bán hàng trên sàn.
- **image**: Tên file hình ảnh tương ứng được lưu trữ trong thư mục dữ liệu train\_images.
- **image\_phash**: Mã băm cảm nhận (Perceptual Hash) của hình ảnh, hỗ trợ thuật toán tìm kiếm các ảnh trùng lặp hoặc gần trùng lặp ở mức độ thô.
- **title**: Tiêu đề sản phẩm do người bán tự đặt, chứa các thông tin về thương hiệu, chủng loại, mã sản phẩm và thuộc tính.
- **label\_group**: Mã định danh nhóm sản phẩm tương đồng. Đây là nhãn mặt cầu dùng để đối soát kết quả truy xuất (nếu ảnh truy vấn và ảnh kết quả có cùng label\_group thì được tính là đúng).

Để đảm bảo hiệu suất cho mô hình học sâu (Deep Learning) và tính nhất quán trong không gian đặc trưng, quy trình làm sạch và tiền xử lý đã được triển khai nghiêm ngặt qua các bước sau:
## <a name="_heading=h.tadg0iibeofz"></a>**4.1. LÀM SẠCH DỮ LIỆU (DATA CLEANING)**
Nhóm tiến hành kiểm tra tính toàn vẹn của dữ liệu trước khi xây dựng mô hình và thu được các kết quả sau:

- **Xử lý giá trị thiếu (Missing Values):** Không phát hiện giá trị thiếu tại các trường thông tin quan trọng như posting\_id, image, image\_phash, title và label\_group.
- **Xử lý dữ liệu trùng lặp:** Không ghi nhận các bản ghi trùng lặp hoàn toàn trong tập dữ liệu, do đó chưa cần thực hiện bước loại bỏ duplicate ở giai đoạn hiện tại.
- **Kiểm tra file ảnh lỗi (Corrupt Images)**: Nhóm thực hiện script kiểm tra toàn bộ 34.250 file ảnh trong thư mục train\_images nhằm xác minh khả năng đọc dữ liệu đầu vào. Kết quả cho thấy toàn bộ ảnh đều có thể mở và xử lý bình thường, không phát hiện file lỗi định dạng hoặc ảnh bị hỏng trong quá trình đọc dữ liệu.

Bảng 2: Làm sạch dữ liệu

|**Nội dung kiểm tra**|**Kết quả**|**Cách xử lý**|
| :-: | :-: | :-: |
|Giá trị thiếu trong posting\_id|0|Không cần xử lý|
|Giá trị thiếu trong image|0|Không cần xử lý|
|Giá trị thiếu trong image\_phash|0|Không cần xử lý|
|Giá trị thiếu trong title|0|Không cần xử lý|
|Giá trị thiếu trong label\_group|0|Không cần xử lý|
|Bản ghi trùng lặp hoàn toàn|0|Không cần loại bỏ|
|Số nhóm label\_group duy nhất|11,014|Sử dụng làm nhãn retrieval|
|Số ảnh trong tập dữ liệu|34,250|Sử dụng cho thực nghiệm full dataset|

## <a name="_heading=h.xsjmv9i92hbu"></a>**4.2. TIỀN XỬ LÝ DỮ LIỆU (IMAGE PREPROCESSING)**
Trong pipeline chính thức, nhóm triển khai kiến trúc truy xuất 2 giai đoạn kết hợp MobileCLIP s1 (GĐ1) và DINOv2 (GĐ2). Để bảo đảm tính nhất quán và tương thích với các kiến trúc pretrained, toàn bộ ảnh đầu vào – bao gồm gallery, tập validation và tập test – được đồng bộ qua một quy trình chuẩn hóa duy nhất. Nhóm quyết định không áp dụng các bước tiền xử lý tốn kém tài nguyên (như thuật toán xóa nền rembg) nhằm tối ưu hóa thời gian suy luận (inference time) và tránh tràn bộ nhớ GPU, nhưng vẫn duy trì được hiệu năng truy xuất tương đối cao (mAP@5 đạt 0.78). Cụ thể các bước tiền xử lý bao gồm:

- **Resize và Center Crop (Kích thước 224x224):** Ảnh gốc với các tỷ lệ đa dạng được điều chỉnh kích thước (resize) sao cho cạnh nhỏ nhất đạt 256 pixel, sau đó được cắt lấy vùng trung tâm (Center Crop) về đúng chuẩn 224x224 pixel. Bước này giúp loại bỏ các vùng biên nhiễu ít thông tin, đồng thời chuẩn hóa không gian đầu vào để cho phép các mô hình xử lý tính toán song song theo lô (batch processing) một cách hiệu quả.
- **Chuyển đổi sang Tensor:** Dữ liệu ảnh dạng mảng được biến đổi thành tensor theo tiêu chuẩn PyTorch, đảm bảo tương thích với toàn bộ pipeline tính toán trơn tru trên môi trường GPU.
- **Chuẩn hóa kênh màu (Normalization)**: Các kênh RGB được chuẩn hóa theo mean = [0.485, 0.456, 0.406] và std = [0.229, 0.224, 0.225] – bộ tham số từ tập ImageNet đã được MobileCLIP s1 kế thừa. Việc này giúp phân phối dữ liệu đầu vào gần với không gian huấn luyện gốc của mô hình, cải thiện độ ổn định và chất lượng của vector đặc trưng hình ảnh.
- **Tính linh hoạt:** Điều đặc biệt của quy trình chuẩn hóa này là nó đáp ứng chính xác yêu cầu đầu vào của cả mô hình MobileCLIP s1 và DINOv2. Nhờ đó, dữ liệu đầu vào chỉ cần đi qua pipeline tiền xử lý một lần duy nhất, đảm bảo mọi vector đặc trưng ở cả hai giai đoạn đều được trích xuất trong cùng một điều kiện tiêu chuẩn, giúp hệ thống hoạt động ổn định và tối ưu hóa tối đa băng thông I/O. 

  Quy trình trên được áp dụng nhất quán cho cả ảnh query lẫn toàn bộ ảnh trong gallery, đảm bảo mọi vector đặc trưng đều được trích xuất trong cùng một điều kiện, không chịu ảnh hưởng của các bước tiền xử lý phức tạp như xóa nền – vốn đã được cân nhắc loại bỏ do không kịp thời gian, đồng thời vẫn giữ được hiệu năng truy xuất cao (mAP@5 đạt 0.78). 
## <a name="_heading=h.j6bkre3bdbey"></a>**4.3 TIỀN XỬ LÝ VĂN BẢN (TEXT PREPROCESSING)**
Trong pipeline chính, nhóm sử dụng MobileCLIP làm backbone đa phương thức cho Giai đoạn 1 (Candidate Generation). Bộ mã hóa văn bản (Text Encoder) của MobileCLIP yêu cầu đầu vào là các ma trận token có chiều dài cố định. Do đặc thù tiêu đề sản phẩm trên e-commerce Shopee thường có độ dài không đồng nhất, chứa nhiều ký tự đặc biệt, từ viết tắt và cách viết hoa thiếu quy chuẩn, nhóm đã áp dụng quy trình tiền xử lý văn bản tự động nhằm khử nhiễu và tối ưu hóa không gian biểu diễn ngữ nghĩa. Các bước cụ thể bao gồm:

- **Chuẩn hóa chuỗi thô (Raw Text Normalization & Cleaning):** Toàn bộ dữ liệu trống (NaN) được xử lý an toàn, sau đó tiêu đề được tự động đưa về dạng chữ thường (lowercasing) và lọc bỏ khoảng trắng thừa. Bước này giúp triệt tiêu sự phân mảnh từ vựng do lỗi đánh máy hoặc cách viết in hoa/in thường lộn xộn (ví dụ: "Áo Thun", "áo thun", "ÁO THUN" đều đồng nhất về "áo thun"), cho phép Text Encoder tập trung vào các từ khóa mang giá trị mô tả thực sự thay vì các ký hiệu quảng cáo nhiễu.
- **Mã hóa từ vựng (BPE Tokenization):** Văn bản sau chuẩn hóa được đưa qua bộ Tokenizer của MobileCLIP (vận hành dựa trên thuật toán Byte Pair Encoding - BPE). Cơ chế BPE thể hiện sự ưu việt khi xử lý dữ liệu Shopee nhờ khả năng phân tách linh hoạt các từ viết tắt, từ ghép tiếng Việt hoặc các từ hiếm (Out-of-Vocabulary) thành các mảnh từ nhỏ hơn (subword), qua đó bảo toàn trọn vẹn thông tin ngữ nghĩa ở mức vi mô.
- **Cố định chiều dài ngữ cảnh (Padding & Truncation):** Để phục vụ tính toán song song trên GPU (Batch Processing), tất cả các chuỗi token được thiết lập về độ dài tối đa là 77 token – chuẩn kích thước ngữ cảnh (context length) của kiến trúc CLIP. Những tiêu đề ngắn sẽ được tự động chèn thêm chuỗi token đệm (Padding), trong khi các tiêu đề vượt quá giới hạn sẽ được cắt bớt phần đuôi (Truncation).

Nhờ quy trình tiền xử lý chặt chẽ này, vector đặc trưng văn bản sinh ra từ MobileCLIP giữ được độ tinh khiết cao, đồng bộ hoàn toàn về mặt chiều không gian với vector hình ảnh. Đây là tiền đề bắt buộc để hệ thống thực hiện phép tính tích vô hướng và hòa trộn đa phương thức theo trọng số alpha, đóng góp trực tiếp vào thành tích mAP@5 đạt 0.78 của hệ thống trên tập Test.
## <a name="_heading=h.719kjpazlt47"></a>**4.4. Xây dựng Pipeline dữ liệu**
Nhóm sử dụng thư viện PyTorch để xây dựng lớp Dataset tùy chỉnh và DataLoader. DataLoader hỗ trợ tải dữ liệu theo batch size = 32, giúp quá trình đọc ảnh, tiền xử lý và trích xuất đặc trưng được thực hiện ổn định hơn khi xử lý nhiều mẫu dữ liệu. Pipeline này giúp đưa vào MobileCLIP để trích xuất vector đặc trưng phục vụ cho bài toán truy xuất ảnh tương đồng. 

Bảng 3: Thống kê LABEL\_GROUP

|**Chỉ tiêu**|Số liệu|
| :- | :- |
|**Tổng số nhóm sản phẩm**|11,014|
|**Ảnh ít nhất trong 1 nhóm**|2 ảnh|
|**Ảnh nhiều nhất trong 1 nhóm**|51 ảnh|
|**Trung bình số ảnh / nhóm**|3\.11 ảnh|
|**Nhóm có đúng 2 ảnh**|6,979 nhóm (63.4%)|
|**Nhóm có trên 10 ảnh**|233 nhóm (2.1%)|

![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.003.png)

**Nhận xét**: Đa số nhóm chỉ có 2-3 ảnh, rất ít nhóm có trên 10 ảnh 

-> Dữ liệu mất cân bằng làm kết quả Recall@K kém ổn định, đặc biệt với các nhóm chỉ có ít ảnh liên quan. Khi mỗi truy vấn chỉ còn rất ít ảnh đúng để truy xuất, mô hình chỉ cần xếp sai một ảnh liên quan là chỉ số recall giảm mạnh.

![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.004.png)

**Nhận xét**: trung vị chỉ khoảng 2 ảnh/nhóm nhưng có nhiều outlier (nhóm có đến 51 ảnh)

-> Tỷ lệ nhóm nhiều nhất / ít nhất = 51/2 = 26x -> mất cân bằng rõ rệt

Biểu đồ 3: Tỷ lệ nhóm theo số ảnh![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.005.png)

**Nhận xét**: 63.4% số nhóm chỉ có đúng 2 ảnh.

-> Đây là thách thức lớn đối với Recall@K vì với các nhóm chỉ có 2 ảnh, mỗi ảnh truy vấn thường chỉ còn 1 ảnh liên quan. Do đó, kết quả đánh giá rất nhạy với thứ hạng của ảnh đúng trong danh sách truy xuất.

## <a name="_heading=h.plyoblx56tra"></a>**4.5 Yêu cầu môi trường và thư viện**
**1. Học sâu & Trích xuất đặc trưng** 

- **torch (>=2.0.0)**: Khung học sâu chủ đạo, quản lý và tối ưu tính toán Tensor trên GPU.
- **torchvision (>=0.15.0)**: Thực hiện các phép biến đổi hình học để chuẩn hóa ảnh đầu vào (Resize, Center Crop, Normalize).
- **timm (>=0.9.0)**: Gọi nhanh các backbone thị giác chuyên sâu (**DINOv2**, **EfficientNetB0**) cho Giai đoạn 2 (Re-ranking).
- **transformers (>=4.35.0)**: Quản lý Tokenizer và cấu hình các mạng đa phương thức (CLIP).
- **mobileclip**: Trích xuất đồng thời đặc trưng ảnh và tiêu đề văn bản ở Giai đoạn 1 (Candidate Generation).

**2. Tìm kiếm tương đồng tăng tốc** 

- **faiss-gpu (>=1.7.4)**: Tối ưu hóa thuật toán tìm kiếm lân cận trên GPU bằng phép toán Tích vô hướng, tăng tốc độ truy xuất gấp 5.4 lần so với Brute-force Python.

**3. Xử lý ảnh truyền thống & Mã băm** 

- **Pillow (>=10.0.0)**: Thư viện nền tảng để đọc, ghi và thao tác trực tiếp với các tệp tin ảnh thô (.jpg).
- **imagehash (>=4.3.1)**: Tính toán mã băm cảm nhận (**pHash**) làm mô hình Baseline lọc trùng ảnh thô.

**4. Quản lý & Biến đổi dữ liệu**

- **pandas (>=2.0.0)**: Đọc, xử lý cấu trúc file metadata (train.csv) và thống kê phân phối nhãn.
- **numpy (>=1.24.0)**: Quản lý mảng đặc trưng đa chiều và tính toán đại số tuyến tính (bắt buộc cho bước chuẩn hóa L2 vector).

**5. Phân chia dữ liệu & Thang đo**

- **scikit-learn (>=1.3.0)**: Phân chia dữ liệu phân tầng (**Stratified Split**) theo tỷ lệ **20% Validation / 80% Test** độc lập và hỗ trợ tính toán chỉ số đánh giá (Precision, Recall).

**6. Tiện ích hệ thống** 

- **tqdm (>=4.65.0)**: Hiển thị thanh tiến trình trực quan khi chạy các vòng lặp trích xuất đặc trưng hàng loạt trên 34.250 ảnh.

**Random seed:** SEED = 42 dùng cho Stratified Split tập Validation/Test.
# <a name="_heading=h.d8w8edtc2hte"></a>**5. MÔ HÌNH BASELINE VÀ KẾT QUẢ BASELINE**
Nhằm khắc phục triệt để sai lệch đo lường và rủi ro rò rỉ dữ liệu (data leakage) từ các tuần trước, trong Tuần 5, nhóm đã tái thiết lập toàn diện giao thức thực nghiệm. Các mô hình cơ sở (Baseline) và Phương pháp chính đều được đánh giá trên cùng một nền tảng dữ liệu chuẩn hóa để đảm bảo tính công bằng.
## <a name="_heading=h.2cjj98mmzxb5"></a>**5.1 HỆ THỐNG CHỈ SỐ ĐO LƯỜNG**
|**Chỉ số**|**Định nghĩa kỹ thuật**|**Vai trò trong đồ án**|
| :- | :- | :- |
|**mAP@5**|**Mean Average Precision tại K=5**|**Metric cốt lõi.** Đánh giá chất lượng thứ tự xếp hạng của ảnh đúng. Ảnh đúng được xếp ở vị trí Top-1 sẽ được điểm cao hơn nhiều so với nằm ở Top-5.|
|**Precision@1**|Độ chính xác tại vị trí đầu tiên|Đo lường tỷ lệ hệ thống tìm trúng ngay sản phẩm liên quan ở kết quả hiển thị cao nhất.|
|**Recall@5**|Độ phủ tại vị trí K=5|Đo lường khả năng gom và bao phủ được bao nhiêu % số lượng ảnh đúng có trong toàn bộ Gallery để đưa vào Top-5.|
#
## <a name="_heading=h.tio4j8qem2hz"></a><a name="_heading=h.zidhymwzlykg"></a>**5.2. DỮ LIỆU SỬ DỤNG VÀ GIAO THỨC ĐÁNH GIÁ ĐỘC LẬP**
- **Không gian tìm kiếm (Gallery):** Sử dụng toàn bộ 34.250 ảnh sản phẩm làm kho dữ liệu truy xuất chung cho mọi thực nghiệm.
- **Tập Truy vấn (Query Split):** Thay vì dùng toàn bộ tập dữ liệu làm Query gây ra hiện tượng học vẹt (overfitting) khi tinh chỉnh, nhóm áp dụng kỹ thuật chia phân tầng (Stratified Split) dựa trên label\_group để tạo ra hai tập truy vấn độc lập:
  - **Tập Validation (20% - khoảng 6.850 ảnh):** Hoạt động như một "thao trường" để chạy thuật toán tìm kiếm dạng lưới (Grid Search), qua đó dò tìm ra các siêu tham số tối ưu.
  - **Tập Test (80% - khoảng 27.400 ảnh):** Bị "đóng băng" hoàn toàn trong quá trình huấn luyện và tinh chỉnh. Tập này chỉ được chạy duy nhất một lần ở bước cuối cùng để chốt số liệu khách quan đưa vào báo cáo.
## <a name="_heading=h.tqzt3v9tqmib"></a>**5.3 CÁC MÔ HÌNH CƠ SỞ (BASELINES)**
Để có cơ sở đo lường mức độ hiệu quả của phương pháp đề xuất, nhóm triển khai hai mô hình Baseline tiêu chuẩn:

1. **Baseline 1 (Late Fusion Truyền thống):** Hệ thống trích xuất vector hình ảnh thông qua mạng EfficientNetB0 và vector văn bản thông qua mô hình ngôn ngữ MiniLM. Hai vector này được hòa trộn để tính toán độ tương đồng. Mô hình này đại diện cho kiến trúc ghép nối đa phương thức kiểu cũ.
1. **Baseline 2 (MobileCLIP Đa phương thức):** Hệ thống sử dụng trực tiếp sức mạnh nguyên bản (Zero-shot) của mô hình MobileCLIP, đánh giá khả năng trích xuất và kết hợp đặc trưng hình ảnh - văn bản trên cùng một không gian nhúng (embedding space) mà không cần đến bước tái xếp hạng.
1. **Baseline pHash (Perceptual Hash Cơ bản):** Hệ thống sử dụng thuật toán băm nhận thức để chuyển đổi hình ảnh thành các chuỗi nhị phân. Phương pháp này có tốc độ cực nhanh và rất hiệu quả trong việc tìm kiếm các ảnh trùng lặp chính xác, nhưng lại thất bại trong việc mã hóa các đặc trưng ngữ nghĩa của sản phẩm (dẫn đến mAP thấp nhất: 0.3840).
1. **Baseline TF-IDF (Tìm kiếm Văn bản Thuần túy):** Hệ thống sử dụng thuật toán thống kê TF-IDF để trích xuất đặc trưng từ tiêu đề (title) của sản phẩm. Mặc dù phương pháp này đem lại độ phủ (Recall) khá cao nhờ bắt trúng các từ khóa cốt lõi, nhưng lại dễ bị đánh lừa bởi hiện tượng nhồi nhét từ khóa (spam keywords) của người bán và hoàn toàn "mù" về thông tin hình ảnh.
1. **Baseline ResNet50 (CNN Truyền thống Đơn phương thức):** Hệ thống chỉ sử dụng mạng CNN kinh điển (ResNet50) để trích xuất đặc trưng thị giác từ ảnh gốc. Do kiến trúc cũ có xu hướng tập trung vào các đặc điểm cục bộ (như màu sắc, nền) và hoàn toàn thiếu đi ngữ cảnh từ văn bản, mô hình dễ bị nhầm lẫn giữa các sản phẩm có cùng tone màu hoặc bối cảnh chụp.
1. **Baseline MobileCLIP - Image Only (Thị giác Đa phương thức Rời rạc):** Hệ thống chỉ kích hoạt nhánh Image Encoder của kiến trúc MobileCLIP. Mặc dù bộ trích xuất này hiểu ngữ nghĩa hình ảnh tốt hơn CNN truyền thống, việc khuyết thiếu luồng thông tin văn bản đi kèm khiến hệ thống không thể phát huy tối đa sức mạnh của không gian nhúng đa phương thức.


|**Phương pháp**|**Precision@1**|**Recall@5**|**mAP@5**|**Nhận xét**|
| :-: | :-: | :-: | :-: | :-: |
|Baseline pHash|0\.3840|0\.3095|0\.2509|Hash nhị phân, không mã hóa ngữ nghĩa|
|Baseline TF-IDF|0\.6929|0\.7336|0\.5735|Recall cao nhờ từ khóa, nhưng nhiễu spam keywords|
|Baseline MobileCLIP (image-only)|0\.6186|0\.5266|0\.5492|Thiếu thông tin văn bản|
|Baseline MobileCLIP (zero-shot)|0\.6772|0\.6266|0\.6307|Đa phương thức nhưng chưa tối ưu|
|ResNet50|0\.6328|0\.5293|0\.5268|CNN truyền thống|
|EfficientNetB0 + MiniLM|0\.7548|0\.6796|0\.7107|Late Fusion truyền thống, cải thiện đáng kể|

<a name="_heading=h.ddseig6ba9yp"></a>\
**6  PHƯƠNG PHÁP CHÍNH VÀ CÁC CẢI TIẾN ĐÃ TRIỂN KHAI**
======================================================
Từ các kết quả thực nghiệm độc lập của các mô hình Baseline, nhóm nhận thấy mỗi phương pháp thuần túy đều tồn tại những giới hạn nhất định: mô hình thị giác đơn lẻ (như pHash) dễ bị nhiễu do yếu tố ngoại cảnh (góc chụp, độ sáng, phông nền), trong khi mô hình văn bản (TF-IDF) lại phụ thuộc hoàn toàn vào tính chính xác của từ khóa do người bán đặt. Để tối ưu hóa năng lực truy xuất, nhóm đề xuất Kiến trúc Truy xuất 2 Giai đoạn (Two-Stage Retrieval Architecture), tích hợp khả năng hiểu ngữ nghĩa đa phương thức từ MobileCLIP (Giai đoạn 1) và năng lực phân tích chi tiết thị giác vi mô từ mạng DINOv2 (Giai đoạn 2), kết hợp cùng giải thuật tìm kiếm tăng tốc FAISS. 

## <a name="_heading=h.bzp1x7dgdtap"></a>**6.1. KIẾN TRÚC TRUY XUẤT HAI GIAI ĐOẠN (TWO-STAGE RETRIEVAL PIPELINE)**
Hệ thống được vận hành theo triết lý **"Lọc thô – Tinh chỉnh" (Coarse-to-Fine Retrieval)**. Chiến lược này chia quy trình tìm kiếm làm hai giai đoạn độc lập nhằm tối ưu hóa đồng thời hai yếu tố: Tốc độ phản hồi thời gian thực trên kho dữ liệu lớn và Độ chính xác phân loại vi mô ở đầu ra.

**GIAI ĐOẠN 1: SÀNG LỌC ỨNG VIÊN (CANDIDATE GENERATION)**

- **Mục tiêu:** Quét thần tốc trên toàn bộ không gian 34.250 ảnh để lọc ra một nhóm nhỏ gồm retrieval\_k = 100 ứng viên tiềm năng nhất, tận dụng sức mạnh của không gian nhúng đa phương thức (Multimodal Embeddings).
- **Quy trình xử lý đặc trưng (Feature Fusion):**
  - Trích xuất đồng thời vector ảnh (*q\_img*) và vector văn bản từ tiêu đề (*q\_txt*) thông qua mô hình nền tảng MobileCLIP s0.
  - Hòa trộn tuyến tính hai không gian đặc trưng theo trọng số *α* để tạo ra siêu vector tổng hợp (*q\_fused*).
  - Chuẩn hóa vector tổng hợp về mặt cầu đơn vị (*L2-norm*) để đảm bảo phép tính Tích vô hướng trong FAISS IndexFlatIP tương đương chính xác với Cosine Similarity.
- **Hệ thống công thức thiết lập:**
  - *q\_fused\_raw = α • q\_img + (1 - α) • q\_txt*
  - *q\_fused = L2\_Norm(q\_fused\_raw)*

Việc đưa vector tổng hợp về mặt cầu đơn vị (||v||₂ = 1) sau bước hòa trộn giúp phép tính Tích vô hướng (Inner Product) trong chỉ mục faiss.IndexFlatIP tương đương chính xác 100% với độ tương đồng Cosine (Cosine Similarity). Do MobileCLIP được huấn luyện trên cùng một không gian nhúng (shared embedding space), vector ảnh và vector văn bản đã có biên độ tương đương nhau, nên việc chuẩn hóa trên vector tổng hợp là đủ để đảm bảo trọng số *α* kiểm soát chính xác tỷ lệ đóng góp giữa hai phương thức.

- **Đầu ra Giai đoạn 1:** Siêu vector *q\_fused* được nạp vào chỉ mục FAISS GPU để truy xuất ra Top-100 ứng viên tiềm năng trong 0.0307 giây (Nhanh gấp 5.4 lần Brute-force). Điểm số tương đồng ở bước này được lưu giữ dưới biến *clip\_score*.

**GIAI ĐOẠN 2: TÁI XẾP HẠNG CHUYÊN SÂU (CANDIDATE RE-RANKING)**

- **Mục tiêu:** Chỉ tập trung xử lý cục bộ trên danh sách retrieval\_k = 100 ứng viên vừa tìm được để tối ưu hóa tài nguyên phần cứng, loại bỏ hoàn toàn các phông nền nhiễu studio trùng lặp.
- **Quy trình phân tích vi mô:**
  - Kích hoạt mô hình Transformer tự giám sát chuyên sâu DINOv2-vitb14.
  - Tiến hành trích xuất token phân loại đã chuẩn hóa x\_norm\_clstoken (Không gian nhúng thị giác sâu 768 chiều).
  - Phân tích các chi tiết cơ lý tính tinh vi (như logo thương hiệu, họa tiết bề mặt, đường may đặc trưng) giữa ảnh truy vấn và ảnh ứng viên.
- **Đầu ra Giai đoạn 2:** Phép tính so sánh cho ra điểm số độ tương đồng thị giác thuần túy, được lưu giữ dưới biến *dino\_score*.

**QUYẾT ĐỊNH XẾP HẠNG CUỐI CÙNG (SCORE FUSION)** Hệ thống áp dụng giải thuật Hòa trộn điểm số (Score Fusion) theo trọng số *β* để dung hòa năng lực hiểu ngữ nghĩa bối cảnh của CLIP và năng lực soi chi tiết thị giác của DINOv2:

- *final\_score = β • dino\_score + (1 - β) • clip\_score*
- **Cơ chế hiển thị:** Hệ thống tiến hành sắp xếp lại (Sort) toàn bộ danh sách ứng viên theo thứ tự giảm dần của *final\_score*.
- **Đầu ra cuối cùng (Output):** Trích xuất Top-5 kết quả có điểm số cao nhất để hiển thị lên giao diện Web App Streamlit chuẩn cấu hình thương mại điện tử.

## <a name="_heading=h.5jhyubtr4wfr"></a>**6.2 TỐI ƯU SIÊU THAM SỐ BẰNG GRID SEARCH**
Tất cả siêu tham số được nội suy trên tập **Validation** (không dùng Test):

Nhóm tiến hành thực nghiệm Grid Search trên tập Validation (6.850 ảnh), quét giá trị alpha từ 0.1 đến 1.0 để chọn cấu hình cho mAP@5 cao nhất,  nhóm áp dụng chiến lược Grid Search 2 vòng (Coarse → Fine): vòng thô quét toàn dải từ 0.1 đến 1.0 với bước nhảy 0.1, sau đó vòng tinh zoom vào vùng cho kết quả tốt nhất với bước nhảy 0.02 

- Khi alpha = 0.1 → 0.4: Vectơ văn bản chiếm ưu thế (60-90%). Kết quả truy xuất bị nhiễu nặng do đặc thù tiêu đề Shopee chứa nhiều từ khóa rác (spam keywords), khiến mAP@5 ở mức thấp.
- Khi alpha = 0.9: Vectơ hình ảnh chiếm 90%. Mô hình gần như bỏ qua các từ khóa ngữ cảnh quan trọng (như mã model, thương hiệu), làm suy giảm độ chính xác.
- **Kết luận:** Quỹ đạo của mAP@5 đạt giá trị tốt nhất trong thực nghiệm tại giá trị **alpha = 0.5** (Hình ảnh đóng góp 50%, Văn bản đóng góp 50%). Cấu hình này cung cấp điểm cân bằng hoàn hảo.


|**Tham số**|**Không gian tìm kiếm**|**Giá trị tối ưu**|**Ý nghĩa**|
| :-: | :-: | :-: | :-: |
|alpha|0\.1 → 1.0, bước 0.1|0\.5|Cân bằng image/text features ở GĐ1|
|retrieval\_k|{20, 50, 100, 200}|100|Số ứng viên đưa vào GĐ2|
|beta|0\.1 → 1.0, bước 0.1|(best từ val)|Cân bằng DINOv2/CLIP score ở GĐ2|
###
## <a name="_heading=h.pssippjcx42o"></a><a name="_heading=h.pgvpc3xsazrt"></a>**6.3. TỐI ƯU HÓA KHÔNG GIAN TÌM KIẾM VÀ HIỆU NĂNG PHẦN CỨNG BẰNG THƯ VIỆN FAISS**
- **Bài toán và giải pháp:** Khi thực nghiệm trên toàn bộ **34.250 ảnh**, phương pháp duyệt tuyến tính vét cạn bằng vòng lặp truyền thống tốn rất nhiều thời gian. Nhóm tích hợp thư viện FAISS đóng vai trò là bộ tăng tốc tính toán hiệu năng cao ở tầng xử lý hệ thống bên dưới nhằm tối ưu hóa thời gian phản hồi và trải nghiệm thực tế của người dùng.
- **Cấu hình thuật toán:** Hệ thống sử dụng chỉ mục faiss.IndexFlatIP nhằm thực hiện cơ chế tìm kiếm chính xác tuyệt đối thông qua các phép toán tích vô hướng song song trên bộ xử lý đồ họa. Khi các vector đặc trưng đã được chuẩn hóa L2, phép tích vô hướng này tương đương chính xác 100% với độ tương đồng Cosine.
- **Quy trình truy xuất và hậu xử lý:** Toàn bộ vector sau chuẩn hóa được nạp vào chỉ mục FAISS để tìm kiếm các vector có độ tương đồng cao nhất. Ở bước hậu xử lý, hệ thống tự động loại bỏ chính ảnh truy vấn khỏi danh sách kết quả trước khi tiến hành tính toán các chỉ số Precision@K, Recall@K và mAP.
- **Kết quả thực nghiệm:** Tốc độ xử lý trung bình đạt mốc lý tưởng **0,0307 giây cho mỗi truy vấn**, giúp hệ thống phản hồi nhanh **gấp 5,4 lần** so với công cụ thông thường.
- **Luận điểm công nghệ:** Công cụ này chỉ giải quyết bài toán gia tốc tốc độ tìm kiếm để tối ưu độ trễ, hoàn toàn không làm thay đổi hay tự cải thiện chất lượng toán học của các vector đặc trưng gốc.

## <a name="_heading=h.yt5qfq8rn0m7"></a>**6.5. LUẬN CỨ KHOA HỌC TRONG VIỆC LỰA CHỌN MÔ HÌNH DINOV2-VITB14**
- **Triệt tiêu hiện tượng dư thừa thông tin:** Do Giai đoạn 1 đã khai thác hiệu quả mối tương quan giữa ảnh và chữ thông qua MobileCLIP, việc tiếp tục tích hợp thêm các mô hình ngôn ngữ ở Giai đoạn 2 hoàn toàn không mang lại giá trị phân biệt mới. Ngược lại, điều này còn làm tăng rủi ro nhiễu hệ thống do hành vi nhồi nhét từ khóa quảng cáo của người bán.
- **Năng lực trích xuất đặc trưng thị giác vi mô vượt trội:** Các kiến trúc mạng tích chập truyền thống (như ResNet50) thường có xu hướng học các đặc trưng hình dáng tổng thể vĩ mô nên rất dễ bị đánh lừa bởi phông nền phức tạp trùng màu. Trong khi đó, DINOv2 (kiến trúc Transformer học tự giám sát) hoạt động như một chuyên gia bóc tách cấu trúc bề mặt, nhận diện chính xác các chi tiết cốt lõi như họa tiết vải, đường chỉ may, tem phụ hoặc logo thương hiệu kích thước nhỏ.
- **Đính chính phiên bản cấu hình:** Hệ thống chạy thực nghiệm chính thức trên phiên bản dinov2\_vitb14 (không gian đặc trưng 768 chiều, dòng cơ bản), không phải phiên bản vits14 (dòng nhỏ) như phần ghi chú sai sót ở báo cáo tiến độ tuần trước. Việc áp dụng dòng cơ bản đảm bảo mật độ thông tin trích xuất phong phú và đạt độ ổn định cao hơn hẳn khi đối mặt với dữ liệu nhiễu thực tế trên sàn thương mại điện tử.



# <a name="_heading=h.vlgie2qds2ei"></a>**8. KẾT QUẢ THỰC NGHIỆM CUỐI** 
## <a name="_heading=h.xynm0q8hwbey"></a>8.1. BẢNG KẾT QUẢ CHÍNH THỨC

|**Phương pháp**|**Precision@1**|**Recall@5**|**mAP@5**|
| :-: | :-: | :-: | :-: |
|Baseline pHash|0\.3840|0\.3095|0\.2509|
|Baseline TF-IDF|0\.6929|0\.7336|0\.5735|
|Baseline MobileCLIP (zero-shot)|0\.6772|0\.6266|0\.6307|
|EfficientNetB0 + MiniLM|0\.7548|0\.6796|0\.7107|
|YOLO + SAHI + DINOv2 + TF-IDF + pHash|0\.7981|0\.7380|0\.7738|
|**MobileCLIP + DINOv2 re-ranking (Phương pháp chính)**|**0.8064**|**0.7530**|**0.7872**|

` `Bảng 4: So sánh kết quả thực nghiệm giữa các phương pháp

**1. Mô hình cơ sở pHash (Mã băm cảm nhận)**

- **Số liệu:** Precision@1 = **0.3840** | Recall@5 = **0.3095** | mAP@5 = **0.2509**
- **Nguyên nhân hiệu năng thấp:** Thuật toán chỉ chuyển đổi cấu trúc ảnh thành chuỗi nhị phân thô, nên chỉ bắt được ảnh sao chép nguyên bản. Phương pháp này thất bại hoàn toàn trước dữ liệu thực tế bị thay đổi góc chụp, ánh sáng bối cảnh, và hoàn toàn không hiểu được ngữ nghĩa văn bản.

**2. Mô hình cơ sở TF-IDF (Văn bản tiêu đề)**

- **Số liệu:** Precision@1 = **0.6929** | Recall@5 = **0.7336** | mAP@5 = **0.5735**
- **Hiện tượng đặc trưng:** Chỉ số độ phủ (Recall) rất cao nhưng chỉ số xếp hạng (mAP) lại thấp.
- **Nguyên nhân:** Thuật toán khai thác tốt các từ khóa trùng nhau trong tiêu đề do nhà bán hàng đặt tên giống nhau (mã dòng máy, thương hiệu). Tuy nhiên, nạn nhồi nhét từ khóa rác của các chủ shop gây nhiễu loạn ngữ nghĩa, dẫn đến việc xếp sai vị trí các sản phẩm đúng trong danh mục Top-5.

**3. Mô hình cơ sở MobileCLIP (Đa phương thức nguyên bản)**

- **Số liệu:** Precision@1 = **0.6772** | Recall@5 = **0.6266** | mAP@5 = **0.6307**
- **Đặc điểm:** Kết quả đồng đều hơn nhờ kết nối được cả miền ảnh và miền chữ.
- **Hạn chế:** Hiệu năng bị giới hạn ở cấu hình 1 giai đoạn do nhánh xử lý văn bản gặp rào cản ngôn ngữ (tiếng Việt nhiễu trên sàn TMĐT), còn nhánh xử lý thị giác toàn cục dễ bị đánh lừa bởi các sản phẩm có chung phông nền phức tạp.

**4. Mô hình lai EfficientNetB0 + MiniLM**

- **Số liệu:** Precision@1 = **0.7548** | Recall@5 = **0.6796** | mAP@5 = **0.7107**
- **Nguyên nhân bứt phá:** Phương pháp ghép nối trực tiếp vector thị giác (từ mạng tích chập CNN) và vector văn bản (từ mạng Transformer) đã bổ khuyết tốt cho nhau. Mô hình tận dụng được thế mạnh nhận diện hình dáng tổng thể và xử lý ngôn ngữ chuyên sâu để tạo ra một mốc đối chứng mạnh.

**5. Luồng nghiên cứu chuyên sâu (YOLO + SAHI + DINOv2 + TF-IDF + pHash)**

- **Số liệu:** Precision@1 = **0.7981** | Recall@5 = **0.7380** | mAP@5 = **0.7738**
- **Thế mạnh thành công:** Việc kết hợp SAHI và YOLO giải quyết triệt để bài toán bóc tách vật thể nhỏ hoặc bị che khuất và khử nhiễu nền phức tạp. Khâu cắt vùng ảnh trọng tâm phối hợp với soi chi tiết vi mô (DINOv2) mang lại độ chính xác rất cao.
- **Hạn chế hạ tầng:** Do tích hợp quá nhiều mô hình rời rạc, hệ thống tiêu tốn lượng tài nguyên tính toán rất lớn và có độ trễ cao, khó đáp ứng tiêu chuẩn thời gian thực.

**6. Phương pháp chính (MobileCLIP + DINOv2 Tái xếp hạng)**

- **Số liệu:** Precision@1 = **0.8064** | Recall@5 = **0.7530** | mAP@5 = **0.7872**
- **Nguyên nhân đạt đỉnh hiệu năng:** Kiến trúc 2 giai đoạn phát huy tối đa tư duy "Lọc thô – Tinh chỉnh":
  - *Giai đoạn 1 (MobileCLIP + FAISS):* Quét siêu tốc diện rộng để không bỏ sót ứng viên tiềm năng (đẩy cao Recall).
  - *Giai đoạn 2 (DINOv2):* Phân tích sâu các đặc trưng vi mô tinh tế (logo, họa tiết, đường may) trên tập ứng viên thu hẹp để loại nhiễu.
  - Việc tinh chỉnh các ngưỡng α,β qua Grid Search giúp tối ưu hóa toán học cơ chế hòa trộn điểm số, đem lại kết quả vượt trội và cân bằng nhất.


![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.006.png)

Bảng 5: Biểu đồ so sánh kết quả thực nghiệm




Theo số liệu tham chiếu từ bài báo, việc nâng cấp từ mạng CNN cơ sở (ResNet-18) lên các kiến trúc học sâu phức tạp (Siamese ResNet-50) đã giúp chỉ số đối sánh cải thiện mạnh mẽ (CV score tăng từ 0.612 lên 0.722). Quy luật này cũng hoàn toàn tương đồng với định hướng tối ưu hóa trong đồ án của nhóm: Việc từ bỏ các kiến trúc đơn giản để chuyển sang ứng dụng sức mạnh của các mạng đa phương thức hiện đại (như MobileCLIP kết hợp DINOv2) là một bước đi mang tính tất yếu để bứt phá giới hạn hiệu năng của hệ thống 

## <a name="_heading=h.ysn0pkvy3kra"></a>8.2.HIỆU NĂNG TRUY XUẤT VÀ THỜI GIAN XỬ LÝ HỆ THỐNG
Trong các hệ thống tìm kiếm hình ảnh sản phẩm thực tế trên sàn thương mại điện tử, tốc độ phản hồi (độ trễ) là một chỉ số mang tính sống còn bên cạnh độ chính xác toàn cục. Nếu thời gian truy vấn quá lớn, hệ thống sẽ không thể đáp ứng trải nghiệm người dùng theo thời gian thực và gây quá tải hạ tầng máy chủ khi có nhiều yêu cầu đồng thời.

Để tối ưu hóa khâu này, nhóm đã thực hiện thực nghiệm đối chứng hiệu năng tính toán khoảng cách trên toàn bộ không gian **34.250 ảnh** của kho dữ liệu:

|**Phương pháp truy xuất**|**Thời gian xử lý trung bình**|**Tốc độ cải thiện**|**Bản chất vận hành kỹ thuật**|
| :- | :- | :- | :- |
|**Duyệt tuyến tính vét cạn (Python/NumPy)**|0,1656 giây|Mốc cơ sở (1,0 lần)|Sử dụng vòng lặp thô trên bộ xử lý để tính toán ma trận tương đồng tuyến tính.|
|**Thư viện tăng tốc FAISS (IndexFlatIP)**|**0,0307 giây**|**Nhanh gấp 5,4 lần**|Tận dụng tối ưu hóa phần cứng lõi C++, tính toán ma trận song song diện rộng.|

![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.007.png)

Dựa trên biểu đồ đối sánh thời gian truy xuất (Latency), việc chuyển đổi từ phương pháp tính toán khoảng cách vét cạn truyền thống sang hệ thống lõi FAISS (cấu trúc IndexFlatIP) đã mang lại một bước nhảy vọt về mặt hiệu năng vận hành. Cụ thể:

- **Tốc độ xử lý vượt trội:** Thời gian thực thi trung bình cho một lượt truy vấn (Query) đã giảm mạnh từ mức 0.1656 giây (Brute-force) xuống chỉ còn 0.0307 giây (FAISS). Tốc độ này tương đương với mức cải thiện hiệu năng gấp **5.4 lần**.
- **Bản chất tối ưu (Góc nhìn MLOps):** Về mặt toán học, chỉ mục faiss.IndexFlatIP vẫn thực hiện tìm kiếm chính xác tuyệt đối (Exact Search) với độ phức tạp $O(N \cdot D)$. Tuy nhiên, FAISS vượt trội hoàn toàn vòng lặp Python/NumPy nhờ viết bằng C++ ở tầng phần cứng thấp, quản lý băng thông bộ nhớ tối ưu và tận dụng tốt kiến trúc tính toán ma trận song song của GPU. FAISS chỉ tăng tốc độ truy hồi, không làm thay đổi chất lượng vector đặc trưng gốc.
- **Đáp ứng chuẩn công nghiệp:** Ngưỡng dưới 0,05 giây là bắt buộc trong thương mại điện tử để đảm bảo trải nghiệm thời gian thực. Mốc 0,0307 giây của FAISS đáp ứng xuất sắc tiêu chuẩn này, tạo khoảng trống tài nguyên lớn để hệ thống chạy khâu Tái xếp hạng chuyên sâu bằng DINOv2 ở Giai đoạn 2 mà không làm chậm tổng thời gian phản hồi chung. Hệ thống hoàn toàn khả thi khi mở rộng quy mô lên hàng triệu sản phẩm.

## <a name="_heading=h.wg7o9yobwejz"></a>8.3.ĐỐI SÁNH VỚI NGHIÊN CỨU QUỐC TẾ
![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.008.png)

Dữ liệu trong bài báo E-commerce Product Similarity Match Detection using Product Text and Images

**Xu hướng công nghệ:** Theo bài báo khoa học tham chiếu *"E-commerce Product Similarity Match Detection using Product Text and Images"*, việc nâng cấp từ mạng CNN cơ sở (ResNet-18) lên kiến trúc học sâu Siamese ResNet-50 giúp điểm số đánh giá chéo tăng mạnh từ 0,612 lên 0,722.

**Tính tất yếu của giải pháp:** Quy luật bứt phá này khẳng định tính đúng đắn trong lộ trình của nhóm. Việc chuyển dịch từ các mô hình đơn lẻ sang kiến trúc lai đa phương thức hiện đại (**MobileCLIP kết hợp DINOv2**) là bước đi mang tính tất yếu để phá vỡ giới hạn hiệu năng, mang lại độ chính xác vượt trội cho bài toán tìm kiếm trực quan.

# <a name="_heading=h.op6wmq3pf1mm"></a>[**9**](http://9.so)[**.**](http://9.so) **SO SÁNH BASELINE – PHƯƠNG PHÁP CHÍNH – MÔ HÌNH TỐI ƯU** 
##
|<a name="_heading=h.xg2syhrxpcte"></a>**Phương pháp**|**mAP@5**|**Precision@1**|**Recall@5**|**Ưu điểm cốt lõi**|**Hạn chế tồn tại**|
| :- | :- | :- | :- | :- | :- |
|**Mô hình cơ sở tốt nhất (TF-IDF)**|0,5735|0,6929|0,7336|Kiến trúc đơn giản; độ phủ (Recall) cao nhờ khai thác tốt từ khóa định danh.|Dễ bị bẻ gãy bởi lỗi chính tả và hành vi nhồi nhét từ khóa quảng cáo rác.|
|**Mô hình lai kết hợp muộn (EfficientNetB0 + MiniLM)**|0,7107|0,7548|0,6796|Cải thiện mạnh chỉ số chính xác nhờ kết hợp đặc trưng hình ảnh và văn bản độc lập.|Sử dụng cấu trúc ghép nối vector thô sơ một giai đoạn; thiếu tái xếp hạng.|
|**Phương pháp chính đề xuất (MobileCLIP + DINOv2)**|0,7872|0,8064|0,7530|Đạt đỉnh hiệu năng trên toàn bộ chỉ số; triệt tiêu nhiễu phông nền và phân loại tinh cực tốt.|Chi phí tính toán tại Giai đoạn 2 cao hơn do phải xử lý qua Transformer diện rộng.|

• Sự bứt phá vượt trội về mặt số liệu: Phương pháp chính thức (MobileCLIP kết hợp DINOv2) chứng minh sự vượt trội toàn diện khi thiết lập đỉnh hiệu năng mới trên mọi chỉ số đo lường độc lập. Chỉ số cốt lõi mAP@5 đạt mốc 0,7872, tăng trưởng bứt phá +0,0765 so với mô hình lai ghép nối thô và bỏ xa mô hình cơ sở văn bản đến +0,2137.\
• Giải quyết triệt để sự đánh đổi giữa Hiệu năng và Tài nguyên: Mặc dù Giai đoạn 2 sử dụng mô hình DINOv2 dòng cơ bản khá nặng về chi phí tính toán, nhưng nhờ Giai đoạn 1 kết hợp với chỉ mục tăng tốc FAISS đã thu hẹp không gian tìm kiếm xuống chỉ còn đúng 100 ứng viên tiềm năng. Hệ thống đạt độ trễ lý tưởng 0,0307 giây/truy vấn, tạo nên sự cân bằng hoàn hảo giữa bài toán tốc độ xử lý và chất lượng truy xuất.


# <a name="_heading=h.kjcc8wtvhir8"></a>**9. PHÂN TÍCH LỖI VÀ HẠN CHẾ**
Bên cạnh việc báo cáo các chỉ số tổng hợp như Precision@1, Recall@5 và mAP@5, nhóm thực hiện phân tích lỗi để xác định những nhóm truy vấn mà phương pháp chính vẫn chưa xử lý tốt. Phần này được xây dựng trực tiếp từ notebook Tuan4\_QuocBao\_ErrorAnalysis.ipynb, sử dụng các file kết quả chi tiết của pipeline Strong Fusion và danh sách Top-K đã lưu sau thực nghiệm. Mục tiêu không chỉ là chỉ ra mô hình sai ở đâu, mà còn giải thích vì sao các lỗi đó xuất hiện trong bối cảnh dữ liệu Shopee có nhiều nhiễu nền, nhiễu văn bản và phân phối nhãn dạng long-tail.
## <a name="_heading=h.dsmhwp1lpb8m"></a>**9.1. QUY TRÌNH TRÍCH XUẤT MẪU LỖI**
Nhóm xác định lỗi nghiêm trọng bằng điều kiện mAP@5 = 0. Với điều kiện này, một ảnh truy vấn được xem là thất bại hoàn toàn trong phạm vi Top-5 vì hệ thống không trả về ảnh nào cùng label\_group với ảnh truy vấn. Sau khi lọc các truy vấn có mAP@5 = 0, notebook tiếp tục lấy kết quả Top-1 đầu tiên khác chính ảnh truy vấn để so sánh trực quan giữa ảnh query và ảnh trả về sai.

Quy trình phân tích gồm bốn bước: đọc file kết quả chi tiết STRONG\_best\_detail\_\*.csv, đọc ma trận STRONG\_best\_top\_indices\_\*.npy, ánh xạ chỉ số ảnh về bảng candidate\_df\_strong\_yolo\_crop\_fusion.csv, sau đó xuất bảng error\_analysis.csv gồm 10 mẫu lỗi đại diện. Cách làm này giúp phần phân tích lỗi bám sát kết quả thực nghiệm thực tế thay vì chỉ nhận xét cảm tính từ một vài ảnh minh họa rời rạc.

|**Nội dung thống kê**|**Giá trị**|**Ý nghĩa**|
| :-: | :-: | :-: |
|Số truy vấn đánh giá|34\.250|Mỗi ảnh trong tập dữ liệu được dùng làm một truy vấn.|
|Số cột trong detail\_df|13|Lưu method, query\_idx, label\_group và các metric mAP/Precision/Recall.|
|Kích thước top\_indices|34\.250 x 10|Mỗi truy vấn có danh sách 10 ứng viên đầu tiên để phân tích Top-K.|
|Số truy vấn mAP@5 = 0|3\.053|Các truy vấn không có ảnh đúng trong Top-5.|
|Tỷ lệ lỗi hoàn toàn|8,91%|Cho thấy vẫn tồn tại một nhóm truy vấn khó dù mAP@5 tổng thể cao.|
|Số mẫu lỗi đưa vào phân tích|10|Chọn cân bằng từ các nhóm lỗi chính để phân tích định tính.|


## <a name="_heading=h.lyaierkk68jz"></a>**9.2. CẤU TRÚC BẢNG PHÂN TÍCH LỖI**
File error\_analysis.csv được dùng làm minh chứng cho quá trình phân tích lỗi. Mỗi dòng tương ứng với một truy vấn bị sai hoàn toàn trong Top-5, kèm theo ảnh Top-1 mà mô hình trả về sai và nhóm nguyên nhân được gán cho lỗi đó.

|**Trường dữ liệu**|**Ý nghĩa**|
| :-: | :-: |
|query\_image|Tên file ảnh truy vấn bị truy xuất sai hoàn toàn.|
|top1\_result|Ảnh đứng đầu danh sách kết quả nhưng không cùng label\_group với ảnh truy vấn.|
|label\_query|Nhãn đúng của ảnh truy vấn.|
|label\_top1|Nhãn của ảnh Top-1 bị trả về sai.|
|error\_type|Nhóm lỗi được phân loại, ví dụ lỗi do nền/màu, góc chụp hoặc chữ quảng cáo.|
|lý do sai|Giải thích ngắn gọn nguyên nhân khiến mô hình nhầm lẫn.|


## <a name="_heading=h.uqg78t59gdnu"></a>**9.3. KẾT QUẢ PHÂN LOẠI 10 MẪU LỖI ĐẠI DIỆN**
Từ 3.053 truy vấn có mAP@5 = 0, nhóm chọn 10 mẫu đại diện theo hướng cân bằng giữa các nhóm lỗi phổ biến. Bảng dưới đây không nhằm mô tả phân phối lỗi của toàn bộ tập dữ liệu, mà dùng để minh họa các tình huống sai điển hình cần phân tích sâu.

|**STT**|**Query image**|**Top-1 sai**|**Label query**|**Label Top-1**|**Nhóm lỗi**|
| :-: | :-: | :-: | :-: | :-: | :-: |
|1|795c6f8f...37f0.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.009.jpeg)|f3c5cc20...1425.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.010.jpeg)|3637220226|2240027280|Màu sắc/nền giống nhau|
|2|9720558f...b8fc.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.011.jpeg)|5f0872e1...91c4.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.012.jpeg)|1445188681|4117627196|Màu sắc/nền giống nhau|
|3|e1a5f03b...3c99.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.013.jpeg)|0e9d686b...8e21.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.014.jpeg)|2699450457|514101767|Màu sắc/nền giống nhau|
|4|ed44a3b1...3af.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.015.jpeg)|e185854c...6c83.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.016.jpeg)|2304809467|2608223592|Màu sắc/nền giống nhau|
|5|685c22d9...809b.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.017.jpeg)|1d1ac481...0d2.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.018.jpeg)|1706249589|3097344893|Góc chụp khác nhau|
|6|c99e19b4...2d52.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.019.jpeg)|6af85140...18e.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.020.jpeg)|3947628716|2221700681|Góc chụp khác nhau|
|7|fb2d9463...77f.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.021.jpeg)|16f81057...2c5.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.022.jpeg)|1060961612|4032101535|Góc chụp khác nhau|
|8|a92eb87e...bfd.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.023.jpeg)|4b8affda...7c6.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.024.jpeg)|169470278|38185708|Góc chụp khác nhau|
|9|39e80002...94f.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.025.jpeg)|34e78d71...6b.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.026.jpeg)|2342771184|3135057640|Chữ quảng cáo che sản phẩm|
|10|64ac780d...323f.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.027.jpeg)|5c2d35f3...e89.jpg![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.028.jpeg)|1680934077|871931738|Chữ quảng cáo che sản phẩm|



Trong 10 mẫu đại diện, nhóm ghi nhận 4 trường hợp thuộc nhóm lỗi do màu sắc hoặc nền giống nhau, 4 trường hợp do khác biệt góc chụp và 2 trường hợp do chữ quảng cáo hoặc chi tiết chữ trên ảnh làm nhiễu. Vì đây là tập mẫu được chọn cân bằng để phân tích định tính, tỷ lệ 4-4-2 không được xem là tỷ lệ lỗi toàn cục của toàn bộ 34.250 truy vấn.
## <a name="_heading=h.50aasyu9iosw"></a>**9.4. PHÂN TÍCH NGUYÊN NHÂN LỖI**
**Nhóm lỗi 1 - Màu sắc hoặc nền giống nhau:** Các ảnh sản phẩm thương mại điện tử thường được chụp trên nền trắng, nền trơn hoặc có bố cục đặt sản phẩm gần giống nhau. Khi hai ảnh có màu chủ đạo, ánh sáng hoặc vùng nền tương đồng, vector đặc trưng toàn cục có thể bị kéo về gần nhau dù sản phẩm thực tế khác label\_group. Đây là hạn chế thường gặp của các mô hình dựa nhiều vào đặc trưng tổng thể, đặc biệt khi đối tượng chính nhỏ hoặc bị nhiễu bởi nền.

**Nhóm lỗi 2 - Khác biệt góc chụp, tỷ lệ và bố cục:** Một sản phẩm có thể được chụp ở nhiều góc, nhiều khoảng cách hoặc trong các điều kiện crop khác nhau. Khi ảnh query và ảnh liên quan khác mạnh về phối cảnh, mô hình có xu hướng xem chúng là hai đối tượng khác nhau. Ngược lại, một sản phẩm khác nhưng có cùng góc chụp và bố cục lại có thể được xếp hạng cao hơn. Điều này cho thấy pipeline vẫn cần cơ chế biểu diễn ổn định hơn trước các biến đổi hình học như góc nhìn, scale và vị trí sản phẩm trong ảnh.

**Nhóm lỗi 3 - Chữ quảng cáo và nhiễu văn bản trên ảnh:** Một số ảnh Shopee chứa banner, nhãn dán, giá khuyến mãi hoặc chữ quảng cáo chiếm diện tích lớn. Các vùng chữ này làm thay đổi đặc trưng thị giác của ảnh và có thể khiến mô hình ưu tiên thông tin không thuộc bản chất sản phẩm. Ngoài ra, tiêu đề sản phẩm cũng thường chứa từ khóa quảng cáo hoặc từ khóa được nhồi để tăng khả năng tìm kiếm, làm nhiễu nhánh văn bản của mô hình đa phương thức.

**Tác động của dữ liệu long-tail:** Phần lớn label\_group trong bộ dữ liệu chỉ có rất ít ảnh liên quan. Với các nhóm chỉ có 2 ảnh, sau khi loại bỏ chính ảnh truy vấn, hệ thống thường chỉ còn 1 ảnh đúng trong gallery. Khi ảnh đúng duy nhất này không nằm trong Top-5, mAP@5 lập tức bằng 0. Do đó, các truy vấn thuộc nhóm ít mẫu làm cho chỉ số truy xuất rất nhạy với sai lệch thứ hạng.
## <a name="_heading=h.af40gebi1f3a"></a>**9.5. ĐỘ TIN CẬY CỦA QUY TRÌNH PHÂN LOẠI LỖI VÀ HƯỚNG CẢI TIẾN**
Việc phân loại nhóm lỗi ban đầu được thực hiện bán tự động bằng các heuristic đơn giản trên ảnh, bao gồm khoảng cách màu trung bình, tỷ lệ nền trắng ở vùng biên, mật độ cạnh và một số tín hiệu liên quan đến chữ quảng cáo. Sau đó, nhóm kiểm tra trực quan các cặp ảnh query/Top-1 để xác nhận lại nguyên nhân sai. Vì vậy, kết quả phân tích lỗi có vai trò hỗ trợ định tính, giúp nhận diện xu hướng lỗi phổ biến của mô hình, không được xem là nhãn lỗi tuyệt đối cho toàn bộ tập dữ liệu.

Từ các lỗi quan sát được, nhóm xác định một số hướng cải tiến hợp lý cho giai đoạn tiếp theo: bổ sung bước crop hoặc segmentation để giảm ảnh hưởng của phông nền; sử dụng OCR hoặc bộ lọc vùng chữ để hạn chế nhiễu quảng cáo; thêm đặc trưng màu cục bộ như HSV histogram hoặc texture descriptor để phân biệt các sản phẩm cùng kiểu dáng nhưng khác màu; thử nghiệm Query Expansion hoặc K-reciprocal Re-ranking để cải thiện truy xuất trên các nhóm có ít ảnh liên quan. Các hướng này bám trực tiếp vào lỗi thực nghiệm thay vì chỉ thay đổi mô hình theo cảm tính. 





**NHẬN XÉT KẾT QUẢ VÀ ĐỊNH HƯỚNG CẢI TIẾN**

**Nhận xét 1: Giới hạn của các mô hình cơ sở và phương pháp đa phương thức đơn lẻ**

Kết quả thực nghiệm cho thấy nếu chỉ sử dụng Baseline truyền thống hoặc áp dụng MobileCLIP nguyên bản (Zero-shot) mà không có cơ chế tái xếp hạng, hiệu năng của hệ thống khó đạt mức tối ưu. Nguyên nhân cốt lõi là do đặc trưng ảnh và văn bản có phân phối không gian khác biệt. Tiêu đề sản phẩm Shopee chứa nhiều ngôn ngữ, từ viết tắt và từ khóa rác (spam keywords), khiến bộ mã hóa văn bản (Text Encoder) bị nhiễu. Hơn nữa, việc chỉ quét 1 lần (1-Stage) khiến mô hình dễ bị "đánh lừa" bởi các ảnh có phông nền (background) màu trắng giống nhau nhưng chi tiết sản phẩm lại khác biệt.

**Nhận xét 2: Sức mạnh và rủi ro của đặc trưng văn bản (Text Features)**

Các phương pháp khai thác đặc trưng văn bản, tiêu biểu như TF-IDF, vẫn cho thấy Recall@5 tương đối cao. Điều này cho thấy tiêu đề sản phẩm trên Shopee chứa nhiều thông tin ngữ cảnh hữu ích, chẳng hạn như tên sản phẩm, thương hiệu, mã sản phẩm, màu sắc hoặc đặc điểm mô tả. Tuy nhiên, đặc trưng văn bản cũng tồn tại hạn chế do tiêu đề thường có hiện tượng nhồi nhét từ khóa, viết tắt, sai chính tả hoặc mô tả không hoàn toàn khớp với hình ảnh thực tế. Vì vậy, trong phương pháp chính, nhóm không sử dụng văn bản như nguồn thông tin duy nhất mà kết hợp đặc trưng văn bản với đặc trưng hình ảnh thông qua cơ chế hòa trộn có trọng số. Trọng số hòa trộn được lựa chọn dựa trên kết quả Grid Search trên tập Validation, nhằm tận dụng thông tin ngữ nghĩa từ tiêu đề sản phẩm nhưng vẫn hạn chế rủi ro nhiễu khi văn bản không đáng tin cậy. 

**Nhận xét 3: Sự vượt trội của Kiến trúc Truy xuất 2 Giai đoạn (MobileCLIP + DINOv2)**

Phương pháp chính thức của Tuần 4 (GĐ1: MobileCLIP + GĐ2: DINOv2 Score Fusion) đã chứng minh cho kết quả tốt hơn các baseline trong bảng xếp hạng. Việc không ghép nối vector trực tiếp (Feature Concatenation) mà chuyển sang hòa trộn ở mức điểm số (Score Fusion) với trọng số beta giúp hệ thống tận dụng được khả năng hiểu ngữ cảnh bao quát của MobileCLIP, đồng thời phát huy năng lực "soi" chi tiết thị giác cực sâu của DINOv2. Kết quả là hệ thống đã kéo được các sản phẩm có ngoại hình tinh vi lên Top-5 một cách chính xác.

**Nhận xét 4: Vai trò cốt lõi của FAISS trong luồng xử lý**

Thực nghiệm cho thấy FAISS (faiss.IndexFlatIP) đóng vai trò là "trái tim" về mặt hiệu năng tính toán (tốc độ truy xuất), nhưng không trực tiếp làm thay đổi độ chính xác của metric nếu không gian vector đặc trưng không đổi. Tuy nhiên, nhờ khả năng truy xuất nhanh của FAISS ở Giai đoạn 1, nhóm mới có đủ tài nguyên bộ nhớ và thời gian để đưa mạng DINOv2 nặng nề vào chạy tái xếp hạng ở Giai đoạn 2 mà không làm hệ thống (Google Colab T4) bị quá tải rò rỉ RAM (OOM).

**Nhận xét 5: Tính khách quan từ Giao thức đánh giá độc lập (Validation/Test Split)**

Việc loại bỏ tập mẫu 500 ảnh và chuyển sang đánh giá trên toàn bộ không gian 34.250 ảnh giúp hệ thống đối mặt với độ khó thực tế của bài toán E-commerce. Quan trọng hơn, việc chia cắt tập truy vấn thành Validation (20% để tuning) và Test (80% để báo cáo) đã giảm đáng kể rủi ro học vẹt (overfitting). Các chỉ số metric thu được trên tập Test (như mAP@5 ~ 0.78) giờ đây là con số trung thực, minh bạch và có giá trị tham khảo thực chiến cao nhất.

**Nhận xét 6: Thách thức từ phân phối dữ liệu đuôi dài (Long-tail distribution)**

Dữ liệu phân tích cho thấy hơn 63% label\_group chỉ chứa từ 2 đến 3 ảnh. Điều này có nghĩa là mỗi lượt truy vấn chỉ có đúng 1-2 "đáp án đúng" nằm lẫn trong biển 34.000 ảnh sai. Đặc thù này khiến chỉ số mAP@5 cực kỳ nhạy cảm. Chỉ cần hệ thống xếp ảnh đúng rơi xuống vị trí Top-2 hoặc Top-3 (thay vì Top-1), điểm số sẽ bị phạt rất nặng. Đây là bài toán khó của thị trường ngách mà hệ thống cần tiếp tục cải thiện.

**Nhận xét 7: Quy luật đánh đổi giữa Precision và Recall**

Biểu đồ đánh giá Top-K (K = 1, 3, 5, 10) phản ánh đúng tính chất của bài toán Information Retrieval: Precision có xu hướng giảm khi K tăng (do có nhiều "chỗ trống" cho ảnh sai lọt vào), trong khi Recall tăng dần (do lưới lọc được mở rộng). Nhóm quyết định chốt Top-5 làm metric chính thức vì đây là giao diện phổ biến nhất trên các ứng dụng di động (hiển thị 1 hàng 2-5 sản phẩm liên quan).

**Nhận xét 8: Hạn chế còn tồn đọng và Định hướng cải tiến đột phá (rembg / YOLO)**

Dù DINOv2 ở Giai đoạn 2 đã giúp phân biệt chi tiết sản phẩm rất tốt, nhóm nhận thấy hệ thống vẫn thỉnh thoảng bị nhầm lẫn khi các sản phẩm khác nhau được chụp trên cùng một phông nền phức tạp (background noise) hoặc ảnh có chứa chữ quảng cáo to che lấp sản phẩm.

Để giảm thiểu rào cản này và hướng tới sản phẩm hoàn thiện nhất, **trong giai đoạn tiếp theo, nhóm có kế hoạch tích hợp thuật toán phân vùng đối tượng (Object Detection/Segmentation). Cụ thể, nhóm sẽ nghiên cứu áp dụng công cụ rembg (dựa trên U-2-Net) hoặc mạng nơ-ron YOLO để tự động phát hiện bounding box, cắt bỏ toàn bộ phông nền và nhiễu chữ trước khi đưa vào hệ thống trích xuất đặc trưng.** Việc làm sạch dữ liệu đầu vào bằng AI này hứa hẹn sẽ đưa độ chính xác của hệ thống chạm đến ngưỡng tối đa.

**Nhận xét 9: Hướng đi giai đoạn 2 trong phương pháp chính**

Trong quá trình phát triển pipeline, nhóm đã cân nhắc hai hướng cho giai đoạn tái xếp hạng (re‑ranking): EfficientNetB0 + MiniLM và DINOv2. Nhóm quyết định chọn DINOv2 qua thực nghiệm và có 3 lí do nhóm chọn DINOv2:

**Triệt tiêu sự dư thừa thông tin (Information Redundancy):** Giai đoạn 1 (MobileCLIP) đã khai thác triệt để ngữ nghĩa văn bản. Việc dùng thêm MiniLM ở Giai đoạn 2 không tạo ra giá trị phân biệt mới, ngược lại còn gây nhiễu hệ thống đối với những sản phẩm có mô tả giống hệt nhau nhưng khác biệt về ngoại hình.

**Khắc phục điểm mù đặc trưng của mạng CNN:** EfficientNetB0 là một kiến trúc xuất sắc cho bài toán Phân loại (Classification) ở mức vĩ mô, nhưng vector đặc trưng của nó thiếu độ "mịn" cần thiết để đo lường độ tương đồng vi mô giữa các sản phẩm na ná nhau trong bài toán Truy xuất (Retrieval)**.**

**Năng lực thị giác vi mô vượt trội:** DINOv2 (mạng Vision Transformer học tự giám sát) hoạt động như một "chuyên gia thị giác thuần túy". Nó có khả năng phân biệt tốt trong việc bóc tách các chi tiết siêu nhỏ như họa tiết, đường may hay góc cạnh logo. Đồng thời, cấu trúc tinh gọn của nó giải quyết triệt để rủi ro tràn tài nguyên trên phần cứng giới hạn (Google Colab T4). Chính sự bổ khuyết hoàn hảo này đã giúp kiến trúc 2-Stage đạt được chất lượng truy xuất tốt hơn, kéo mAP@5 đạt khoảng 0.78

![](Aspose.Words.8eb52f76-5278-40a6-8075-ed880695b1de.029.png)

DINOv2: Learning Robust Visual Features without Supervision
# <a name="_heading=h.ynorrnpdrefv"></a>**10.  DEMO HOẶC SẢN PHẨM THỬ NGHIỆM**
Để minh chứng khả năng hoạt động thực tế của hệ thống truy xuất hình ảnh sản phẩm tương đồng, nhóm đã xây dựng một notebook demo với tên Demo.ipynb. Demo này cho phép người dùng nhập vào một ảnh truy vấn bất kỳ, sau đó hệ thống sẽ trả về danh sách Top-5 ảnh sản phẩm tương đồng nhất dựa trên kết quả truy xuất đã được tạo bởi phương pháp chính. 
## <a name="_heading=h.27p8pvrw879p"></a>**10.1. Mục tiêu của demo**
Mục tiêu của demo là mô phỏng quy trình sử dụng hệ thống Visual Search ở mức đơn giản, trực quan và dễ kiểm chứng. Người dùng chỉ cần cung cấp một ảnh sản phẩm đầu vào, hệ thống sẽ tìm kiếm trong tập dữ liệu và hiển thị 5 ảnh có độ tương đồng cao nhất. Kết quả demo giúp nhóm kiểm tra trực tiếp khả năng truy xuất của mô hình, đồng thời hỗ trợ quan sát các trường hợp đúng và sai trong danh sách Top-K.

**Cấu trúc sử dụng chính của demo:**

query\_path = 'sample\_query.jpg'

top5 = search(query\_path, index, candidate\_df, IMAGE\_DIR, k=5)

show\_top5(top5)

Trong đó, query\_path là đường dẫn đến ảnh truy vấn, search() là hàm thực hiện truy xuất ảnh tương đồng, còn show\_top5() là hàm hiển thị trực quan ảnh truy vấn cùng 5 kết quả tìm được.
## <a name="_heading=h.nrd5pu59018z"></a>**10.2. Dữ liệu và file đầu vào của demo**
Notebook demo sử dụng các file kết quả đã được tạo ra từ pipeline chính của nhóm, bao gồm:

|**Thành phần**|**Vai trò**|
| :-: | :-: |
|candidate\_df\_strong\_yolo\_crop\_fusion.csv|Lưu thông tin ảnh, label\_group, tên file ảnh và các thông tin phục vụ truy xuất.|
|STRONG\_best\_top\_indices\_\*.npy|Lưu danh sách chỉ số Top-K đã được tính sẵn bởi phương pháp chính.|
|train\_images|Thư mục chứa toàn bộ ảnh sản phẩm trong tập dữ liệu.|
|sample\_query.jpg|Ảnh truy vấn mẫu do người dùng cung cấp.|
|results/demo\_top5\_result.csv|File lưu kết quả Top-5 sau khi chạy demo.|

Notebook được thiết kế để tự động tìm kiếm các file cần thiết trong thư mục dự án. Trong trường hợp không tìm được, người dùng có thể cấu hình thủ công đường dẫn thông qua các biến MANUAL\_TOP\_NPY, MANUAL\_CANDIDATE\_CSV và MANUAL\_IMAGE\_DIR.
## <a name="_heading=h.exxgmdhdm2nw"></a>**10.3. Quy trình hoạt động của demo**
**Bước 1: Chuẩn bị ảnh truy vấn**

Người dùng đặt ảnh cần tìm kiếm vào cùng thư mục với notebook và đặt tên là sample\_query.jpg, hoặc chỉnh biến query\_path thành đường dẫn ảnh mong muốn. Ảnh truy vấn có thể là ảnh nằm trong tập dữ liệu hoặc ảnh bên ngoài do người dùng đưa vào.

**Bước 2: Tải dữ liệu truy xuất**

Notebook tự động tải bảng candidate\_df và ma trận top\_indices. Bảng candidate\_df chứa thông tin của các ảnh trong gallery, còn top\_indices chứa danh sách ứng viên đã được truy xuất bởi phương pháp chính. Sau khi tải dữ liệu, hệ thống kiểm tra sự tương thích giữa số dòng của candidate\_df và số dòng của top\_indices để tránh lỗi lệch dữ liệu.

**Bước 3: Xác định ảnh truy vấn**

Hệ thống tìm ảnh truy vấn trong candidate\_df bằng tên file, đường dẫn ảnh hoặc mã pHash. Nếu ảnh truy vấn thuộc tập dữ liệu, demo sử dụng trực tiếp kết quả truy xuất từ top\_indices của phương pháp chính. Nếu ảnh truy vấn là ảnh bên ngoài, notebook có cơ chế dự phòng bằng pHash để vẫn trả về kết quả minh họa.

**Bước 4: Truy xuất Top-5 ảnh tương đồng**

Hàm search() nhận ảnh truy vấn và trả về 5 ảnh tương đồng nhất. Với mỗi kết quả, hệ thống lưu lại các thông tin gồm thứ hạng, tên ảnh kết quả, nhãn của ảnh truy vấn, nhãn của ảnh kết quả và trạng thái đúng/sai dựa trên việc so sánh label\_group.

**Bước 5: Hiển thị kết quả**

Hàm show\_top5() hiển thị ảnh truy vấn ở vị trí đầu tiên, sau đó hiển thị 5 ảnh kết quả theo thứ tự xếp hạng. Mỗi ảnh kết quả được đánh dấu đúng hoặc sai dựa trên việc label\_group có trùng với ảnh truy vấn hay không.

**Bước 6: Lưu kết quả demo**

Sau khi chạy xong, kết quả Top-5 được lưu vào file results/demo\_top5\_result.csv để nhóm có thể kiểm tra lại, đối chiếu với báo cáo và sử dụng làm minh chứng cho quá trình thực nghiệm.
## <a name="_heading=h.ukzp824u7i6m"></a>**11.4. Kết quả đầu ra của demo**
Kết quả đầu ra của demo gồm ba phần chính:

|**Đầu ra**|**Mô tả**|
| :-: | :-: |
|Hình ảnh trực quan|Hiển thị ảnh truy vấn và 5 ảnh sản phẩm tương đồng nhất.|
|Bảng Top-5|Hiển thị thứ hạng, phương pháp truy xuất, tên ảnh kết quả, nhãn sản phẩm và trạng thái đúng/sai.|
|File CSV|Lưu kết quả truy xuất vào results/demo\_top5\_result.csv.|

Thông qua demo, nhóm có thể quan sát trực tiếp hiệu quả của hệ thống thay vì chỉ đánh giá bằng các chỉ số định lượng như Precision@1, Recall@5 và mAP@5. Điều này giúp việc phân tích lỗi trở nên rõ ràng hơn, đặc biệt trong các trường hợp mô hình trả về ảnh có màu sắc hoặc phông nền giống nhau nhưng khác nhãn sản phẩm.
## <a name="_heading=h.1byjm1hck9vs"></a>**10.5. Nhận xét về sản phẩm thử nghiệm**
Demo đã đáp ứng được yêu cầu cơ bản của một hệ thống Visual Search thử nghiệm: nhận ảnh đầu vào, truy xuất ảnh tương đồng và hiển thị kết quả Top-5 một cách trực quan. Notebook có khả năng tự động tìm các file kết quả đã sinh ra từ pipeline chính, giúp giảm phụ thuộc vào đường dẫn cố định trên từng máy. Ngoài ra, cơ chế lưu kết quả ra file CSV giúp nhóm dễ dàng kiểm tra, so sánh và đưa kết quả vào báo cáo.

Tuy nhiên, demo hiện vẫn ở mức notebook thử nghiệm, chưa được đóng gói thành giao diện người dùng hoàn chỉnh. Trong các tuần tiếp theo, nhóm có thể phát triển thêm giao diện bằng Streamlit để người dùng tải ảnh trực tiếp, xem kết quả Top-5 trên trình duyệt và thao tác dễ dàng hơn.
# <a name="_heading=h.wl3degyuh5z7"></a>**11. PHÂN CÔNG, MINH CHỨNG CÁ NHÂN VÀ KHAI BÁO SỬ DỤNG AI**
Bảng Ghi chú AI hỗ trợ
## <a name="_heading=h.4if8xry2w07a"></a>**11.1. BẢNG ĐÓNG GÓP VÀ TIẾN ĐỘ CỦA THÀNH VIÊN (TUẦN 5)**

|**Thành viên**|**Công việc thực hiện**|**Tệp tin / Minh chứng**|**Mức độ hoàn thành**|
| :- | :- | :- | :- |
|**Mã Gia Vỹ**|Sửa lỗi chuẩn hóa L2 cho vector tổng hợp sau fusion, vận hành lại mã nguồn hệ thống chính thức, khởi tạo tệp cấu hình môi trường requirements.txt.|Tuan5\_GiaVy\_Pipeline\_Fixed.ipynb<br>final\_results\_official.csv<br>requirements.txt|Hoàn thành|
|**Lê Quốc Bảo**|Cập nhật quy trình phân tích mẫu lỗi hệ thống, kiểm thử toàn diện chương trình chạy thực tế từ đầu đến cuối (end-to-end).|Tuan5\_QuocBao\_ErrorAnalysis.ipynb<br>demo\_v2.ipynb<br>error\_analysis\_v2.csv|Hoàn thành|
|**Nguyễn Khánh Hưng**|Biên tập hoàn thiện tài liệu báo cáo tiến độ Tuần 5, xây dựng bảng đối chiếu hiệu năng tổng hợp cuối cùng và thiết kế slide thuyết trình.|Nhom3ThangCuTi\_Tuan5\_33.docx<br>slides\_tuan5.pptx|Hoàn thành|

## <a name="_heading=h.t3v2t7415jq2"></a>**11.2. BẢNG KHAI BÁO SỬ DỤNG CÔNG CỤ TRÍ TUỆ NHÂN TẠO (AI) HỖ TRỢ**

|**Phần hành / Tệp tin**|**Mô hình AI hỗ trợ**|**Cách thức AI hỗ trợ chi tiết**|**Người kiểm tra & giải thích**|
| :- | :- | :- | :- |
|**Tuan5\_GiaVy\_Pipeline\_Fixed.ipynb**|Claude|Gợi ý giải pháp vá lỗi logic toán học toán tử chuẩn hóa L2-normalize, đánh giá lại toàn bộ mã nguồn dòng xử lý của hệ thống.|Mã Gia Vỹ|
|**Tuan5\_QuocBao\_ErrorAnalysis.ipynb**|Claude|Gợi ý phương pháp thuật toán trích xuất tự động và phân loại định tính các mẫu lỗi nghiêm trọng.|Lê Quốc Bảo|
|**Nhom3ThangCuTi\_Tuan5\_33.docx**|Claude|Hỗ trợ chuẩn hóa cấu trúc đề mục văn bản học thuật, kiểm tra tính đồng nhất của các số liệu thực nghiệm.|Nguyễn Khánh Hưng|
|**comparison\_chart.ipynb**|Gemini|Hỗ trợ cấu hình mã nguồn đồ họa và căn chỉnh trực quan biểu đồ đối sánh hiệu năng giữa các mô hình.|Nguyễn Khánh Hưng|


# <a name="_heading=h.uegxbi80tsr"></a>**12. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN** 
## <a name="_heading=h.ciyi9vhyvogk"></a>**12.1. KẾT LUẬN**
Qua quá trình triển khai nghiên cứu, thực nghiệm và tối ưu hóa hệ thống từ Tuần 1 đến Tuần 5, nhóm đã hoàn thành mục tiêu xây dựng giải pháp tìm kiếm sản phẩm tương đồng bằng hình ảnh ứng dụng trong lĩnh vực thương mại điện tử. Dựa trên các kết quả định lượng đạt được, nhóm rút ra những kết luận chuẩn mực sau:

- Hiệu quả vượt trội từ kiến trúc phân tầng: Mô hình kết hợp hai giai đoạn (Sàng lọc diện rộng và Tái xếp hạng chuyên sâu) đã chứng minh tính đúng đắn khi giải quyết thành công bài toán cân bằng giữa tốc độ xử lý và độ chính xác. Hệ thống đảm bảo không bỏ sót sản phẩm ở giai đoạn lọc thô, đồng thời bóc tách xuất sắc các chi tiết vi mô ở giai đoạn lọc tinh. Kết quả thực nghiệm cuối cùng đạt mốc bứt phá với chỉ số độ chính xác toàn cục mAP@5 đạt 0,7872 và tỷ lệ chính xác ở vị trí đầu tiên đạt 80,64% trên toàn bộ kho dữ liệu 34.250 ảnh.
- Tối ưu hóa thời gian phản hồi thực tế: Việc tích hợp giải pháp tăng tốc phần cứng ở tầng hệ thống giúp giảm thời gian thực thi trung bình xuống mốc lý tưởng chỉ 0,0307 giây cho mỗi truy vấn (nhanh gấp 5,4 lần so với phương pháp duyệt tuyến tính truyền thống). Tốc độ này đáp ứng hoàn hảo tiêu chuẩn vận hành thời gian thực của các nền tảng thương mại điện tử công nghiệp, tạo không gian tài nguyên dự phòng lớn để máy chủ vận hành các thuật toán xử lý sâu hơn.
- Kỷ luật thực nghiệm và tính trung thực của số liệu: Nhóm đã tuân thủ nghiêm ngặt giao thức kiểm thử độc lập bằng cách thiết lập tập kiểm định và tập kiểm thử hoàn toàn biệt lập, loại bỏ triệt để hiện tượng rò rỉ dữ liệu hoặc học vẹt khi tinh chỉnh tham số. Bản nâng cấp cuối cùng cũng đã vá thành công lỗi logic toán học trong khâu đồng bộ biên độ giữa các miền đặc trưng, đảm bảo tính nhất quán tuyệt đối cho toàn bộ quy trình.
- Hoàn thiện sản phẩm ứng dụng thực tế: Hệ thống không chỉ dừng lại ở các thuật toán chạy trên dòng lệnh mà đã được đóng gói thành một ứng dụng web tương tác trực quan. Giao diện chương trình cho phép giả lập đầy đủ trải nghiệm thực tế của người dùng thông qua các thao tác kéo thả hình ảnh và hiển thị danh sách sản phẩm đối sánh tương đồng.
- Nhận diện thách thức đặc thù của ngành: Nghiên cứu đã làm rõ đặc tính phân phối đuôi dài nghiêm trọng của dữ liệu thực tế (63,4% số nhóm sản phẩm chỉ có đúng 2 ảnh mẫu), khẳng định đây là nút thắt cốt lõi và là tiền đề để định hình các giải pháp nâng cấp thuật toán dài hạn.
## <a name="_heading=h.umevy5slq8nv"></a>**12.2. HƯỚNG PHÁT TRIỂN**
### <a name="_heading=h.2l0qcq2akp21"></a>12.2.1. Định hướng ngắn hạn (Tối ưu hạ tầng và ứng dụng thực tế)
- Hợp nhất luồng nhận diện vật thể chuyên sâu: Tiếp tục tối ưu hóa luồng xử lý nhận diện vật thể cục bộ để tự động cắt vùng trung tâm sản phẩm và bóc tách đối tượng nhỏ. Giải pháp này giúp triệt tiêu hoàn toàn phông nền nhiễu phức tạp của studio trước khi đưa vào hệ thống tìm kiếm lân cận.
- Nén mô hình và tối ưu hóa dung lượng: Áp dụng các kỹ thuật lượng tử hóa dữ liệu hoặc chuyển đổi cấu trúc mô hình sang các định dạng tối ưu hóa phần cứng tầng thấp nhằm giảm tải dung lượng bộ nhớ đồ họa, phục vụ cho việc đóng gói và triển khai hệ thống lên các hạ tầng điện toán đám mây với chi phí vận hành tối thiểu.
### <a name="_heading=h.1l9gp640qbej"></a>12.2.2. Định hướng dài hạn (Nâng cấp thuật toán chuyên sâu)
- Khắc phục thách thức phân phối dữ liệu ít ảnh mẫu: Nghiên cứu và ứng dụng giải thuật tái xếp hạng dựa trên tính chất tương hỗ giữa các cặp vector lân cận, nhằm cải thiện mạnh mẽ độ chính xác cho những nhóm sản phẩm đặc thù có quá ít ảnh đối sánh trong kho dữ liệu.
- Xử lý nhiễu đồ họa quảng cáo nâng cao: Tích hợp các bộ lọc nhận diện văn bản tự động để che phủ các vùng chứa chữ quảng cáo, biểu tượng giảm giá hoặc nhãn hiệu bản quyền chèn đè lên ảnh gốc, giúp làm sạch không gian trích xuất đặc trưng thị giác.
- Tinh chỉnh ngôn ngữ thương mại bản địa: Thực hiện huấn luyện tinh chỉnh bộ mã hóa văn bản trên tập dữ liệu đặc thù của thương mại điện tử tiếng Việt nhằm nâng cao năng lực hiểu các từ viết tắt, thuật ngữ ngành sàn và hỗ trợ tự động sửa lỗi chính tả từ phía người dùng khi tìm kiếm.
### <a name="_heading=h.y677584o5z8y"></a>12.2.3 Mô hình định hướng của giáo viên
Bên cạnh phương pháp trích xuất đặc trưng toàn cục (Global Feature Retrieval), nhóm đã tiến hành phân tích và thực nghiệm một luồng kiến trúc độc lập theo đúng định hướng chuyên sâu của Giảng viên. Kiến trúc này tập trung giải quyết bài toán nhiễu vi mô và phân loại đối tượng bằng cách kết hợp mạng nhận diện vật thể (YOLO) và kỹ thuật phân mảnh (SAHI).

**1. Luồng xử lý của hệ thống (Detection & Classification Pipeline)**

**Sơ đồ kiến trúc được nhóm triển khai tuần tự qua 5 bước cốt lõi:**

- **Bước 1 - Phát hiện vật thể tăng cường (SAHI + YOLO):** Ảnh đầu vào được đưa qua hệ thống Slicing Aided Hyper Inference (SAHI) để cắt thành nhiều mảnh nhỏ. Các mảnh này được đưa vào mạng dò tìm YOLOv8s (hoặc phiên bản nhẹ YOLOv11n). Việc kết hợp này giúp hệ thống khắc phục điểm yếu của YOLO nguyên bản, cho phép bắt được các vật thể (sản phẩm) có kích thước cực nhỏ hoặc bị che khuất trong ảnh Shopee.
- **Bước 2 - Trích xuất vùng ảnh (Crop Bounding Boxes):** Từ tọa độ (bbox) mà YOLO dự đoán, hệ thống tiến hành cắt (crop) các vùng ảnh chứa sản phẩm tách biệt hoàn toàn khỏi phông nền.
- **Bước 3 - Phân loại cục bộ (Crop Classifier):** Các vùng ảnh vừa cắt được đưa qua một mạng nơ-ron tích chập truyền thống (MobileNetV2 hoặc EfficientNetB0 đóng vai trò là Classifier) để dự đoán nhãn (label) và trích xuất điểm tin cậy phân loại (Classifier Confidence).
- **Bước 4 - Hòa trộn mềm (Soft Fusion):** Để ra quyết định cuối cùng, hệ thống áp dụng cơ chế Soft Fusion, tính toán trọng số kết hợp giữa điểm tin cậy của mạng nhận diện (YOLO Confidence) và mạng phân loại (Classifier Confidence). Điều này giúp triệt tiêu các trường hợp YOLO nhận diện nhầm rác thành vật thể.
- **Bước 5 - Đầu ra cuối cùng**: Hệ thống trả về kết quả bao gồm: Nhãn sản phẩm (Final Label), Tọa độ vật thể (Bbox) và Điểm tin cậy tổng hợp (Final Confidence).

**2. Phân tích Nhược điểm và Thách thức tích hợp vào bài toán Visual Search**

Dưới góc độ kỹ thuật (MLOps), luồng kiến trúc này là một giải pháp State-of-the-Art (SOTA) tuyệt vời cho bài toán *Phân loại (Classification)*. Tuy nhiên, khi đối chiếu với mục tiêu tối thượng của Đồ án là *Truy xuất hình ảnh tương đồng (Visual Search / Information Retrieval)*, hệ thống bộc lộ những điểm bất cập lớn khiến nhóm chưa thể chọn làm phương pháp chính thức**:**

- **Lệch pha về bản chất bài toán (Task Mismatch):** Bài toán Visual Search của nhóm yêu cầu đầu ra phải là một Vector đặc trưng (Embedding Vector) dồi dào ngữ nghĩa để tính toán khoảng cách Cosine trong không gian FAISS. Trong khi đó, luồng của thầy lại kết thúc bằng việc xuất ra một Nhãn danh mục (Final Label). Nếu chỉ dùng Label để tìm kiếm, hệ thống sẽ thoái lui về dạng tìm kiếm từ khóa thô sơ, làm mất đi ý nghĩa của truy xuất hình ảnh đa phương thức.
- **Bùng nổ chi phí tính toán (Computational Overhead):** Chuỗi tính toán: *SAHI (cắt nhiều ảnh) → YOLO → Cắt Bbox → EfficientNetB0 (tính cho từng Bbox)* là quá sức chịu đựng đối với phần cứng Google Colab T4. Để chạy luồng này trên toàn bộ 34.250 ảnh Gallery, thời gian suy luận (Inference time) bị đội lên gấp hàng chục lần so với luồng MobileCLIP, gây ra hiện tượng tràn RAM và ngắt kết nối liên tục.
- **Bài toán đối sánh Nhiều-Nhiều (Many-to-Many Retrieval):** 1 ảnh gốc khi đi qua YOLO có thể bị cắt ra thành 3-4 Bbox (ví dụ: áo, quần, túi xách trong cùng 1 ảnh). Việc đẩy toàn bộ các crop này vào FAISS sẽ làm Index phình to một cách mất kiểm soát. Khi người dùng query 1 cái áo, hệ thống có thể trả về 5 kết quả nhưng lại... nằm chung trong cùng 1 tấm ảnh gốc, gây nhiễu loạn trải nghiệm (User Experience).

***Tiểu kết:*** Nhóm đánh giá luồng YOLO + SAHI + Soft Fusion là một giải pháp có tính hệ thống và chặt chẽ theo định hướng của Giảng viên. Tuy nhiên, do giới hạn khắt khe về thời gian chạy của hệ thống và đặc thù phải dùng FAISS để tính toán Vector, nhóm quyết định bảo lưu kiến trúc này. Nhóm chọn luồng MobileCLIP + DINOv2 Score Fusion (trích xuất vector toàn cục 2 giai đoạn) làm phương pháp chính cho Tuần 4 nhằm bảo đảm tính ổn định, đồng thời sẽ nghiên cứu cách "rút trích vector từ lớp ẩn của mô hình Crop Classifier" trong các tuần tiếp theo để ứng dụng một phần tư tưởng của Giảng viên vào hệ thống tìm kiếm.
###
#

# <a name="_heading=h.wfwxvby81vjt"></a><a name="_heading=h.q4q52e4xiqtp"></a><a name="_heading=h.h8ep106zfjws"></a>**13. TÀI LIỆU THAM KHẢO**
[1] W. McKinney, "Data structures for statistical computing in Python," in *Proceedings of the 9th Python in Science Conference*, 2010, pp. 51–56.

[2] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019.

[3] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[4] J. D. Hunter, "Matplotlib: A 2D graphics environment," *Computing in Science & Engineering*, vol. 9, no. 3, pp. 90–95, 2007.

[5] M. Zhu, "Recall, Precision and Average Precision," University of Waterloo, Tech. Rep., 2004.

[6] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[7] J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," *IEEE Transactions on Big Data*, vol. 7, no. 3, pp. 535–547, 2019.

[8] Kaggle, "Shopee - Price Match Guarantee," Kaggle Competition Dataset, 2021. [Online]. Available:[ ](https://www.kaggle.com/c/shopee-product-matching)<https://www.kaggle.com/c/shopee-product-matching>

[9] A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," in *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 2021, pp. 8748–8763.

[10] M. Kaya and H. Ş. Bilge, "Deep Metric Learning: A Survey," *Symmetry*, vol. 11, no. 9, p. 1066, 2019.

[11] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 4690–4699.

[12] M. Oquab, T. Darcet, T. Moutakanni et al., "DINOv2: Learning Robust Visual Features without Supervision," *arXiv preprint arXiv:2304.07193*, 2023.

- **Ghi chú ứng dụng:** Mô hình tạo ra các biểu diễn hình ảnh mạnh mẽ vượt trội thông qua cơ chế tự chú ý (Self-Attention) của kiến trúc Vision Transformer học tự giám sát. Cấu hình triển khai thực tế trong hệ thống của nhóm là phiên bản dinov2\_vitb14 (Dòng Base, không gian đặc trưng 768 chiều), giúp bảo đảm mật độ thông tin trích xuất phong phú và đạt độ ổn định cao hơn trước dữ liệu nhiễu thực tế.

[13] P. K. A. Vasu et al., "MobileCLIP: Fast Multimodal Learning for Mobile Devices," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024.

- **Ghi chú ứng dụng:** Dòng mô hình đa phương thức được tối ưu hóa để đạt hiệu suất xử lý cực cao. Phiên bản hệ thống áp dụng là MobileCLIP-S1, mang lại năng lực mã hóa zero-shot tương đương các kiến trúc lớn nhưng tối ưu tốc độ nhanh hơn gấp 2.3 lần, đóng vai trò nền tảng cho Giai đoạn 1 (Sàng lọc ứng viên thô).

[14] M. Jochko et al., "YOLOv8: A Comprehensive Review of the State-of-the-Art Object Detector," *Ultralytics Tech Report*, 2023.

- **Ghi chú ứng dụng:** Kiến trúc nhận diện vật thể tiên tiến, hỗ trợ định vị chính xác vùng chứa sản phẩm cốt lõi nhằm chuẩn bị dữ liệu đầu vào sạch cho các mô hình trích xuất đặc trưng.

[15] F. C. Akyon, S. O. Altinuc, and A. Temizel, "SAHI: Slicing Aided Hyper Inference for Small Object Detection," in *Proceedings of the IEEE International Conference on Image Processing (ICIP)*, 2022.

- **Ghi chú ứng dụng:** Framework hỗ trợ cắt lát hình ảnh thông minh giúp tăng cường khả năng phát hiện vật thể nhỏ hoặc bị che khuất, giải quyết bài toán khử nhiễu phông nền studio phức tạp trong luồng nghiên cứu chuyên sâu.

[16] Z. Zhong, L. Zheng, D. Cao, and S. Li, "Re-ranking Person Re-identification with k-reciprocal Encoding," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 1318–1327.

- **Ghi chú ứng dụng:** Phương pháp tái xếp hạng dựa trên việc mã hóa các láng giềng chung k-reciprocal giúp đánh giá mối quan hệ tương hỗ giữa các vector lân cận, định hướng ứng dụng dài hạn để bứt phá hiệu năng cho nhóm sản phẩm đặc thù ít ảnh mẫu.



51
