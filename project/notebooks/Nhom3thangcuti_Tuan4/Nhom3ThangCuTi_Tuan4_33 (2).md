
![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.001.png)

**UBND THÀNH PHỐ HỒ CHÍ MINH**

**TRƯỜNG ĐẠI HỌC SÀI GÒN**


![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.002.png)
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

[12. TÀI LIỆU THAM KHẢO	35](#_heading=h.189v3ox24ggx)

#
# <a name="_heading=h.vupppftv6dve"></a>**
# Bảng phân công

|**MSSV**|**Họ và tên**|**Nhiệm vụ trong tuần**|
| :-: | :-: | :-: |
|3124410015|Lê Quốc Bảo|Phân tích lỗi chuyên sâu, chạy thử mô hình theo hướng của giảng viên, chạy demo|
|3124410414|Mã Gia Vỹ|Thêm 2 baseline MobileCLIP và EfficientNetB0 + MiniLM để so sánh, test một vài mô hình và chốt pipeline kết hợp các mô hình hiện đại  |
|3124410129|Nguyễn Khánh Hưng|Lập bảng thống kê, viết kế hoạch tuần 5|
# <a name="_heading=h.t67ldu7xrd87"></a>**Danh mục từ viết tắt**

|**Từ viết tắt**|**Thuật ngữ đầy đủ**|**Ý nghĩa**|
| :-: | :-: | :-: |
|CNN|Convolutional Neural Network|Mạng nơ-ron tích chập, dùng để trích xuất đặc trưng từ hình ảnh.|
|EDA|Exploratory Data Analysis|Phân tích dữ liệu khám phá, bước tìm hiểu đặc tính của bộ dữ liệu.|
|FAISS|Facebook AI Similarity Search|Thư viện tối ưu hóa việc tìm kiếm các vector tương đồng.|
|mAP|Mean Average Precision|Độ chính xác trung bình (chỉ số chính để đánh giá hệ thống truy vấn).|
|Recall@K|Recall at K|Độ phủ tại vị trí K (tỷ lệ ảnh đúng tìm thấy trong top K kết quả).|
|ResNet|Residual Network|Một kiến trúc mạng nơ-ron sâu (Deep Learning) phổ biến.|
|Embedding|Feature Vector|Vector đặc trưng đại diện cho nội dung của một bức ảnh dưới dạng số.|
|Baseline|Baseline Model|Mô hình cơ sở dùng làm mốc so sánh.|


# <a name="_heading=h.qqip2smcx6tu"></a>1. TÓM TẮT VÀ MỤC TIÊU 
## <a name="_heading=h.s78ob1qay0d"></a> 1.1 TÓM TẮT

|**Nội dung**|**Chi tiết thông tin**|
| :-: | :-: |
|**Mã đề tài/nhóm**|Mã đề tài: 33- Visual Search, Nhóm: 3 thằng cu tí|
|**Loại bài toán**|Truy xuất hình ảnh|
|**Dữ liệu sử dụng**|<p>Tên: Shopee - Price Match Guarantee</p><p>Nguồn: Kaggle Competition Dataset</p><p>Số lượng mẫu: 34,250</p>|
|**Phương pháp tuần này**|<p>MobileCLIP với trọng số fusion alpha được tối ưu bằng grid search trên tập validation làm giai đoạn 1, và DINOv2 tái xếp hạng (re‑ranking) kết hợp cơ chế fusion điểm số Score Fusion (với trọng số beta cùng số lượng ứng viên retrieval\_k được tối ưu tương tự) ở giai đoạn 2 nhằm tái xếp hạng ứng viên và trả về Top-5 kết quả có độ tương đồng cao nhất. </p><p>Song song với phương pháp chính, nhóm cũng đã triển khai thực nghiệm một luồng kiến trúc độc lập bám sát theo định hướng của Giảng viên, tập trung vào khâu tiền xử lý hình ảnh chuyên sâu (triệt tiêu phông nền và nhận diện vật thể vi mô). Đáng chú ý, hướng tiếp cận này đã mang lại kết quả tích cực ngoài mong đợi khi hệ thống ghi nhận mức hiệu năng mAP@5 đạt 0.77. </p>|
|**Metric chính**|Precision@K, Recall@K, mAP|
|**Kết quả** |Phương pháp MobileCLIP (GĐ1) + DINOv2(GĐ2) đã đạt hiệu quả hơn đa số baseline. Tuy nhiên, hướng cải tiến YOLO + SAHI + Box + MobileCLIP + DINOv2 sẽ được nhóm tiếp tục tối ưu và triển khai trong Tuần 5 tới.|

*Báo cáo này trình bày quá trình nghiên cứu và phát triển hệ thống truy xuất hình ảnh sản phẩm tương đồng (Visual Search) trên bộ dữ liệu Shopee - Price Match Guarantee, bao gồm 34.250 ảnh thuộc 11.014 nhóm sản phẩm. Kết quả phân tích dữ liệu ban đầu cho thấy bộ dữ liệu có hiện tượng mất cân bằng rõ rệt theo phân phối đuôi dài (long-tail), trong đó phần lớn các nhóm sản phẩm chỉ chứa từ 2 đến 3 ảnh. Ở Tuần 2, nhóm đã xây dựng mô hình cơ sở (baseline) sử dụng MobileCLIP để trích xuất đặc trưng hình ảnh. Tuy nhiên, việc đánh giá trên tập con ngẫu nhiên 500 ảnh chưa đảm bảo mỗi ảnh truy vấn đều có ảnh cùng nhóm trong không gian tìm kiếm, khiến các chỉ số đo lường như Precision@K, Recall@K và mAP chưa phản ánh đầy đủ hiệu năng thực tế của hệ thống.*

*Tiếp thu góp ý từ Giảng viên, trong Tuần 3 nhóm đã điều chỉnh không gian tìm kiếm (Gallery) lên toàn bộ 34.250 ảnh, đồng thời tích hợp thư viện FAISS nhằm tối ưu hóa tốc độ truy xuất trên dữ liệu lớn. Các thực nghiệm với pHash, TF-IDF và MobileCLIP đã được tiến hành. Dù vậy, phương pháp ghép nối đặc trưng đa phương thức ở tuần này chưa đạt hiệu quả như kỳ vọng do khoảng cách ngữ nghĩa giữa đặc trưng ảnh và văn bản, cũng như đặc thù tiêu đề sản phẩm Shopee chứa nhiều ngôn ngữ và từ khóa nhiễu. Thêm vào đó, việc tinh chỉnh tham số trực tiếp trên tập đánh giá chung đã bộc lộ rủi ro rò rỉ dữ liệu (data leakage) và học vẹt (overfitting).*

*Khắc phục triệt để các hạn chế trên, trong Tuần 4, nhóm đã thiết lập giao thức đánh giá chuẩn mực bằng cách chia tập truy vấn (Query) thành tập Validation (20%) và tập Test (80%) độc lập. Phương pháp chính thức được nhóm đề xuất là kiến trúc truy xuất 2 giai đoạn (2-Stage Retrieval). Cụ thể, Giai đoạn 1 (Candidate Generation) ứng dụng MobileCLIP để kết hợp đặc trưng đa phương thức thông qua trọng số alpha, kết hợp tìm kiếm FAISS để trích xuất tập ứng viên tiềm năng (retrieval\_k). Bước sang Giai đoạn 2 (Re-ranking), hệ thống sử dụng khả năng trích xuất đặc trưng thị giác sâu của mạng DINOv2. Thông qua cơ chế Score Fusion, điểm độ tương đồng của DINOv2 được hòa trộn tuyến tính với điểm số MobileCLIP ban đầu theo trọng số beta. Toàn bộ các siêu tham số (alpha, beta, retrieval\_k) đều được tinh chỉnh tối ưu (Grid Search) nghiêm ngặt trên tập Validation trước khi áp dụng để tái xếp hạng ứng viên, qua đó giúp hệ thống giảm ảnh hưởng của nhiễu phông nền và trả về Top-5 kết quả chính xác nhất trên tập Test.*

*Song song với kiến trúc chính, nhóm cũng đã mở rộng thực nghiệm một luồng xử lý độc lập bám sát định hướng của Giảng viên. Bằng việc áp dụng các kỹ thuật Computer Vision chuyên sâu để làm sạch dữ liệu (xóa phông nền, bóc tách vật thể nhỏ), luồng tiếp cận này đã mang lại kết quả tích cực ngoài mong đợi với mAP@5 = 0.77, qua đó khẳng định tiềm năng to lớn của việc tích hợp Object Detection vào bài toán tìm kiếm trực quan.* 
## <a name="_heading=h.wly463n1si88"></a>1.2 MỤC TIÊU
**Trong phạm vi báo cáo Tuần 4, kế thừa và khắc phục các hạn chế từ Tuần 3, mục tiêu trọng tâm của nhóm được xác định như sau:**

- **Tiền xử lý và phân tích dữ liệu (EDA):** Chuẩn hóa dữ liệu đầu vào, kiểm tra tính toàn vẹn và phân tích hiện tượng mất cân bằng nhóm sản phẩm (phân phối đuôi dài - long-tail) trên 11.014 label\_group của bộ dữ liệu Shopee.
- **Cập nhật và chuẩn hóa giao thức đánh giá:** Khắc phục triệt để rủi ro rò rỉ dữ liệu (data leakage) gặp phải ở Tuần 3. Nhóm thiết lập giao thức đánh giá mới: thiết lập toàn bộ 34.250 ảnh làm Không gian tìm kiếm (Gallery), phân chia tập Truy vấn (Query) thành tập Validation (20% - dành riêng cho việc tinh chỉnh tham số) và tập Test (80% - đánh giá độc lập). Việc này giúp phản ánh khách quan và minh bạch nhất hiệu năng truy xuất thực tế của hệ thống.
- **Xây dựng các mô hình cơ sở (Baseline):** Đánh giá lại ba phương pháp cơ sở trên giao thức mới, bao gồm: pHash (mã băm cấu trúc ảnh), TF-IDF (đặc trưng văn bản từ tiêu đề) và MobileCLIP (đặc trưng đa phương thức Zero-shot). Các Baseline này đóng vai trò làm mốc tiêu chuẩn (benchmark) khắt khe để đo lường mức độ đột phá của phương pháp chính.
- **Triển khai kiến trúc truy xuất 2 giai đoạn (2-Stage Retrieval):** Phát triển và tối ưu hóa hệ thống phương pháp chính. Giai đoạn 1 (Candidate Generation) ứng dụng MobileCLIP kết hợp thư viện FAISS để quét nhanh trên không gian dữ liệu lớn, lọc ra tập ứng viên tiềm năng (retrieval\_k). Giai đoạn 2 (Re-ranking) sử dụng mô hình DINOv2 để phân tích chi tiết hình ảnh sâu, kết hợp cơ chế Score Fusion nhằm hòa trộn điểm số giữa DINOv2 và MobileCLIP. Toàn bộ các siêu tham số (alpha cho Giai đoạn 1, beta và retrieval\_k cho Giai đoạn 2) đều được tìm kiếm dạng lưới (Grid Search) trên tập Validation.
- **So sánh đa mô hình và phân tích lỗi chuyên sâu:** Đánh giá đối chiếu toàn diện các phương pháp thông qua bộ chỉ số chuẩn: mAP@5 (metric chính), Precision@1 và Recall@5. Dựa trên kết quả, nhóm tiến hành bóc tách các trường hợp truy xuất sai (như lỗi trùng màu phông nền, nhiễu văn bản) để chứng minh tính hiệu quả vượt trội của mạng DINOv2 trong khâu tái xếp hạng.

# <a name="_heading=h.ag07gxu4psdz"></a>**2. LỊCH SỬ LÀM VIỆC TRONG TUẦN 4**
<a name="_heading=h.rzbnt0veceti"></a>BẢNG PHÂN CÔNG

|**Thời điểm**|**Thành viên thực hiện**|**Nội dung công việc**|**Sản phẩm/minh chứng**|**Trạng thái**|
| :-: | :-: | :-: | :-: | :-: |
|26/5-27/5|Mã Gia Vỹ|<p>Chạy baseline 1: EfficientNetB0 + MiniLM</p><p>Chạy baseline 2: </p><p>MobileCLIP</p><p></p>|<p>Baseline1\_EfficientNetB0\_MiniLM.ipynb</p><p>Baseline2\_MobileCLIP.ipynb</p><p></p>|Hoàn thành|
|27/5 - 29/5|<p>Nguyễn Khánh Hưng</p><p></p><p></p><p></p><p></p><p></p><p>Mã Gia Vỹ</p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p>Lê Quốc Bảo</p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p>Cả nhóm</p><p></p>|<p>Lập bảng thống kê, viết kế hoạch tuần 5</p><p></p><p></p><p></p><p></p><p>Chạy pipeline</p><p>MobileCLIP + DINOv2 + re-ranking + grid search alpha</p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p>Chạy phương pháp định hướng của giảng viên, demo</p><p></p><p></p><p></p><p></p><p></p><p></p><p>Tổng hợp và cùng nhau viết báo cáo</p>|<p>Tuan4\_KhanhHung\_MetricUnify.ipynb</p><p>comparison\_chart.ipynb</p><p></p><p></p><p></p><p>Tuan4\_GiaVy\_Pipeline.ipynb, final\_metric.csv,</p><p>final\_metric\_DINOv2.csv</p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p>Tuan4\_StrongFusion\_YOLOCrop\_DINOv2\_TFIDF\_pHash.ipynb</p><p>demo.ipynb</p><p></p><p></p><p></p><p></p><p></p><p>Nhom3ThangCuTi\_Tuan4\_33.docx</p>|<p>Hoàn thành</p><p></p><p></p><p></p><p></p><p></p><p></p><p>Hoàn thành</p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p>Hoàn thành</p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p>Đang cải thiện</p>|
|30/5|Cả nhóm|Tổng hợp và hoàn thành báo cáo |Nhom3ThangCuTi\_Tuan4\_33.docx|Hoàn thành|

# <a name="_heading=h.vkcrki525b8e"></a>**3. CÁC VẤN ĐỀ CÒN TỒN TẠI TỪ TUẦN 3 VÀ CÁCH ĐÃ SỬA**

|<h2></h2>|
| :- |
||
|<p><h2><a name="_heading=h.cpxgo36z2ffo"></a><a name="_heading=h.r4d1c19eonh9"></a>**3.1 Minh bạch hóa quy trình tinh chỉnh siêu tham số:**</h2></p><p>Thay vì thiết lập cứng một giá trị trọng số theo cảm tính, nhóm đã bổ sung mục phân tích chi tiết thuật toán Tìm kiếm dạng lưới (Grid Search). Các siêu tham số cốt lõi bao gồm alpha (trọng số hòa trộn đa phương thức ở Giai đoạn 1) và beta (trọng số Score Fusion ở Giai đoạn 2) đều được chứng minh bằng biểu đồ biến thiên của thang đo mAP@5, đảm bảo tính chặt chẽ về mặt khoa học.</p><p><h2><a name="_heading=h.cibjw8irpsns"></a>**3.2 Cụ thể hóa nền tảng toán học trong thuật toán Tái xếp hạng:** </h2></p><p>Nhóm đã cấu trúc lại một mục riêng để chuẩn hóa các công thức tính toán của hệ thống. Trong đó, cơ chế pHashBoost cũ đã được thay thế hoàn toàn bằng công thức Hòa trộn điểm số tuyến tính (Linear Score Fusion), cho phép hợp nhất điểm tương đồng ngữ nghĩa của MobileCLIP và độ lệch thị giác vi mô của DINOv2 một cách mượt mà và phù hợp hơn. Nhóm sẽ xem xét đưa pHashBoost và các phương pháp xử lý phông nền do thầy cung cấp nếu có thời gian.</p><p><h2><a name="_heading=h.mhngnw23dwss"></a>**3.3 Triển khai giao thức đánh giá độc lập (Tránh Data Leakage):**</h2></p><p>Nhóm đã tái cấu trúc toàn diện quy trình kiểm thử bằng kỹ thuật chia phân tầng (Stratified Split). Dữ liệu truy vấn được phân tách nghiêm ngặt thành tập **Validation (20%)** chuyên dụng làm "thao trường" dò tìm tham số và tập **Test (80%)** độc lập để chốt sổ báo cáo cuối cùng. Bước đi này giảm đáng kể rủi ro học vẹt (overfitting) do việc tuning trực tiếp trên tập Test ở tuần trước.</p>|
||
||
# <a name="_heading=h.johpkjwd3thl"></a>**4. DỮ LIỆU VÀ PIPELINE XỬ LÝ CUỐI CÙNG**
Bảng 1: Thống kê tổng quan bộ dữ liệu

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

Nhờ quy trình tiền xử lý chặt chẽ này, vector đặc trưng văn bản sinh ra từ MobileCLIP giữ được độ tinh khiết cao, đồng bộ hoàn toàn về mặt chiều không gian với vector hình ảnh. Đây là tiền đề bắt buộc để hệ thống thực hiện phép tính tích vô hướng (Cosine Similarity) và hòa trộn đa phương thức (Fusion) theo trọng số $\alpha$, đóng góp trực tiếp vào thành tích mAP@5 đạt 0.78 của hệ thống trên tập Test.
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

![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.003.png)

**Nhận xét**: Đa số nhóm chỉ có 2-3 ảnh, rất ít nhóm có trên 10 ảnh 

-> Dữ liệu mất cân bằng làm kết quả Recall@K kém ổn định, đặc biệt với các nhóm chỉ có ít ảnh liên quan. Khi mỗi truy vấn chỉ còn rất ít ảnh đúng để truy xuất, mô hình chỉ cần xếp sai một ảnh liên quan là chỉ số recall giảm mạnh.

![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.004.png)

**Nhận xét**: trung vị chỉ khoảng 2 ảnh/nhóm nhưng có nhiều outlier (nhóm có đến 51 ảnh)

-> Tỷ lệ nhóm nhiều nhất / ít nhất = 51/2 = 26x -> mất cân bằng rõ rệt

Biểu đồ 3: Tỷ lệ nhóm theo số ảnh![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.005.png)

**Nhận xét**: 63.4% số nhóm chỉ có đúng 2 ảnh.

-> Đây là thách thức lớn đối với Recall@K vì với các nhóm chỉ có 2 ảnh, mỗi ảnh truy vấn thường chỉ còn 1 ảnh liên quan. Do đó, kết quả đánh giá rất nhạy với thứ hạng của ảnh đúng trong danh sách truy xuất.
# <a name="_heading=h.d8w8edtc2hte"></a>**5. MÔ HÌNH BASELINE VÀ PHƯƠNG PHÁP CHÍNH ĐÃ TRIỂN KHAI**
Nhằm khắc phục triệt để sai lệch đo lường và rủi ro rò rỉ dữ liệu (data leakage) từ các tuần trước, trong Tuần 4, nhóm đã tái thiết lập toàn diện giao thức thực nghiệm. Các mô hình cơ sở (Baseline) và Phương pháp chính đều được đánh giá trên cùng một nền tảng dữ liệu chuẩn hóa để đảm bảo tính công bằng.
## <a name="_heading=h.zidhymwzlykg"></a>**5.1. DỮ LIỆU SỬ DỤNG VÀ GIAO THỨC ĐÁNH GIÁ ĐỘC LẬP**
- **Không gian tìm kiếm (Gallery):** Sử dụng toàn bộ 34.250 ảnh sản phẩm làm kho dữ liệu truy xuất chung cho mọi thực nghiệm.
- **Tập Truy vấn (Query Split):** Thay vì dùng toàn bộ tập dữ liệu làm Query gây ra hiện tượng học vẹt (overfitting) khi tinh chỉnh, nhóm áp dụng kỹ thuật chia phân tầng (Stratified Split) dựa trên label\_group để tạo ra hai tập truy vấn độc lập:
  - **Tập Validation (20% - khoảng 6.850 ảnh):** Hoạt động như một "thao trường" để chạy thuật toán tìm kiếm dạng lưới (Grid Search), qua đó dò tìm ra các siêu tham số tối ưu.
  - **Tập Test (80% - khoảng 27.400 ảnh):** Bị "đóng băng" hoàn toàn trong quá trình huấn luyện và tinh chỉnh. Tập này chỉ được chạy duy nhất một lần ở bước cuối cùng để chốt số liệu khách quan đưa vào báo cáo.
## <a name="_heading=h.tqzt3v9tqmib"></a>**5.2. CÁC MÔ HÌNH CƠ SỞ (BASELINES)**
Để có cơ sở đo lường mức độ hiệu quả của phương pháp đề xuất, nhóm triển khai hai mô hình Baseline tiêu chuẩn:

1. **Baseline 1 (Late Fusion Truyền thống):** Hệ thống trích xuất vector hình ảnh thông qua mạng EfficientNetB0 và vector văn bản thông qua mô hình ngôn ngữ MiniLM. Hai vector này được hòa trộn để tính toán độ tương đồng. Mô hình này đại diện cho kiến trúc ghép nối đa phương thức kiểu cũ.
1. **Baseline 2 (MobileCLIP Đa phương thức):** Hệ thống sử dụng trực tiếp sức mạnh nguyên bản (Zero-shot) của mô hình MobileCLIP, đánh giá khả năng trích xuất và kết hợp đặc trưng hình ảnh - văn bản trên cùng một không gian nhúng (embedding space) mà không cần đến bước tái xếp hạng.
## <a name="_heading=h.ddseig6ba9yp"></a>**5.3. KIẾN TRÚC PHƯƠNG PHÁP CHÍNH: TRUY XUẤT 2 GIAI ĐOẠN (2-STAGE RETRIEVAL)**
Khắc phục nhược điểm của các Baseline, nhóm đề xuất kiến trúc hệ thống 2 giai đoạn nhằm tối ưu hóa cả ngữ nghĩa tổng thể lẫn chi tiết thị giác vi mô.

- **Giai đoạn 1 (Candidate Generation - Quét diện rộng):** Đặc trưng hình ảnh và văn bản của ảnh truy vấn được trích xuất qua MobileCLIP. Hai vector này được hòa trộn theo công thức tuyến tính: q\_fused = L2\_Norm(α \* q\_img + (1-α) \* q\_txt). Vector kết quả được đưa vào hệ thống FAISS (faiss.IndexFlatIP) để tính toán Tích vô hướng (Inner Product) với toàn bộ 34.250 ảnh Gallery trong thời gian thực, từ đó lọc ra danh sách ứng viên tiềm năng (Top-retrieval\_k).
- **Giai đoạn 2 (Re-ranking - Tái xếp hạng bằng DINOv2):** Danh sách ứng viên tiếp tục được đưa qua mạng DINOv2 (phiên bản vitb14) để trích xuất đặc trưng thị giác cực sâu. Điểm tương đồng DINOv2 (dino\_score) giữa ảnh truy vấn và ứng viên được tính bằng Cosine Similarity. Cuối cùng, hệ thống áp dụng cơ chế Score Fusion: final\_score = β \* dino\_score + (1-β) \* clip\_score để sắp xếp lại độ ưu tiên và lọc ra Top-5 ứng viên có độ tương đồng cao nhất. Các trọng số alpha, beta và retrieval\_k đều được nội suy từ tập Validation.
## <a name="_heading=h.rytt5k9lf1a"></a>**5.4. HỆ THỐNG CHỈ SỐ ĐÁNH GIÁ (METRICS)**
- **Tiêu chí đúng/sai:** Một ứng viên trong danh sách trả về được tính là kết quả Đúng (True Positive) nếu trường label\_group của nó trùng khớp hoàn toàn với label\_group của ảnh truy vấn.
- **Các bộ chỉ số đo lường:** Hiệu năng của toàn bộ hệ thống được chốt bằng 3 chỉ số chuyên dụng cho bài toán truy xuất: **Precision@1 (độ chính xác của kết quả đứng đầu)**, **Recall@5** (khả năng bao phủ các ảnh liên quan trong Top-5), và **mAP@5** (Mean Average Precision - chỉ số cốt lõi đánh giá chất lượng thứ tự xếp hạng).
- **Loại bỏ tự truy xuất (Self-Similarity Removal):** Trong mọi giao thức test, hệ thống được cấu hình tự động nhận diện và loại bỏ chính bản thân ảnh truy vấn (thông qua query\_idx) khỏi danh sách kết quả, đảm bảo hệ thống thực sự "tìm kiếm" các sản phẩm tương đồng chứ không phải "trả bài" dữ liệu đầu vào.
## <a name="_heading=h.urakrqu5k7b9"></a>**5.5. MÔ HÌNH ĐỊNH HƯỚNG CỦA GIÁO VIÊN**
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
# <a name="_heading=h.lholqsmq2o9l"></a>**6.  CÁC CẢI TIẾN/ TỐI ƯU TRONG TUẦN 4:**
Từ các kết quả thực nghiệm độc lập của các mô hình Baseline, nhóm nhận thấy mỗi phương pháp thuần túy đều tồn tại những giới hạn nhất định: mô hình thị giác đơn lẻ (như pHash) dễ bị nhiễu do yếu tố ngoại cảnh (góc chụp, độ sáng, phông nền), trong khi mô hình văn bản (TF-IDF) lại phụ thuộc hoàn toàn vào tính chính xác của từ khóa do người bán đặt. Để tối ưu hóa năng lực truy xuất, nhóm đề xuất Kiến trúc Truy xuất 2 Giai đoạn (Two-Stage Retrieval Architecture), tích hợp khả năng hiểu ngữ nghĩa đa phương thức từ MobileCLIP (Giai đoạn 1) và năng lực phân tích chi tiết thị giác vi mô từ mạng DINOv2 (Giai đoạn 2), kết hợp cùng giải thuật tìm kiếm tăng tốc FAISS. 
## <a name="_heading=h.41pnq3weoaq4"></a>**6.1. KIẾN TRÚC HỆ THỐNG 2 GIAI ĐOẠN VÀ CƠ SỞ LÝ LUẬN**
Kiến trúc đề xuất được thiết kế dựa trên triết lý "Lọc thô - Tinh chỉnh" (Coarse-to-Fine Retrieval) thường được ứng dụng trong các hệ thống E-commerce quy mô lớn.

- **Giai đoạn 1 (Candidate Generation - Quét diện rộng):** Hoạt động dựa trên không gian nhúng đa phương thức (Multimodal Embedding Space). Bằng cách hòa trộn hai chiều không gian (Hình ảnh và Văn bản), mô hình có khả năng tương hỗ thông tin: ngữ nghĩa của tiêu đề sản phẩm sẽ bù đắp cho những sai lệch về mặt thị giác của ảnh. Hệ thống ưu tiên tốc độ, lọc ra một tập nhỏ các sản phẩm tiềm năng nhất.
- **Giai đoạn 2 (Re-ranking - Tái xếp hạng):** Đóng vai trò như một "kính lúp" soi chiếu chi tiết. Giai đoạn này sử dụng mạng thị giác tự giám sát sâu (Self-Supervised Vision Transformer) để soi xét các ứng viên, triệt tiêu các lỗi sai do trùng màu nền hoặc sai biệt chi tiết nhỏ mà Giai đoạn 1 bỏ sót, từ đó đưa ra quyết định xếp hạng cuối cùng.
## <a name="_heading=h.z6cvazqty8ul"></a>**6.2. QUY TRÌNH TRÍCH XUẤT, HÒA TRỘN VÀ TÁI XẾP HẠNG (PIPELINE CHÍNH THỨC)**
- **Bước 1: Trích xuất và Hòa trộn Đa phương thức (MobileCLIP - GĐ1)**\
  Ảnh truy vấn và tiêu đề sản phẩm sau khi đi qua pipeline tiền xử lý sẽ được chuyển vị qua mạng MobileCLIP s1. Hệ thống đồng thời trích xuất Vector Hình ảnh và Vector Văn bản . Thay vì kết hợp đặc trưng bằng phép ghép nối (Concatenation) làm bùng nổ số chiều tính toán, nhóm áp dụng công thức **Hòa trộn Tuyến tính (Linear Feature Fusion)** theo trọng số alpha đã được nội suy từ tập Validation:\
  `        `***![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.006.png)***\
  Việc chuẩn hóa L2 (L2-Normalization) ngay sau khi hòa trộn giúp đưa độ dài Euclid của vector về mặt cầu đơn vị (||v||₂ = 1), đảm bảo sự ổn định khi tính toán độ tương đồng.
- **Bước 2: Tìm kiếm Tập ứng viên với FAISS (Candidate Generation)**\
  Vector đa phương thức q\_fused được đưa vào cấu trúc chỉ mục faiss.IndexFlatIP (chứa dữ liệu của 34.250 ảnh Gallery). Nhờ đặc tính của L2-Norm, phép toán Tích vô hướng (Inner Product) lúc này tương đương chính xác với Cosine Similarity. FAISS quét qua toàn bộ dữ liệu trong thời gian thực và trả về danh sách Top-K ứng viên tiềm năng (retrieval\_k). Đồng thời, hệ thống ghi nhận lại điểm số ban đầu của các ứng viên này làm cơ sở (clip\_score).
- **Bước 3: Trích xuất Đặc trưng Thị giác Sâu (DINOv2 - GĐ2)**\
  Tại bước này, hệ thống chỉ tập trung tính toán trên danh sách retrieval\_k ứng viên vừa được lọc ra (thay vì toàn bộ 34 ngàn ảnh, giúp triệt tiêu rủi ro tràn RAM GPU). Ảnh truy vấn và các ảnh ứng viên được đưa qua mạng DINOv2\_vitb14. Hệ thống trích xuất token phân loại đã được chuẩn hóa (x\_norm\_clstoken), thu được Vector Thị giác sâu (q\_dino) kích thước 768 chiều. Từ đó, tính toán độ tương đồng giữa ảnh truy vấn và ứng viên (dino\_score).
- **Bước 4: Cơ chế Tái xếp hạng bằng Hòa trộn Điểm số (Score Fusion)**\
  Thay vì bỏ rơi thông tin ngữ nghĩa từ Giai đoạn 1, nhóm thiết kế thuật toán **Hòa trộn Điểm số (Score Fusion)** để hợp nhất "tư duy tổng quan" của MobileCLIP và "độ sắc nét" của DINOv2. Điểm số xếp hạng cuối cùng (final\_score) được tính toán bằng cách hòa trộn theo trọng số beta (đã được tối ưu qua Grid Search):\
  ![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.007.png)Hệ thống tiến hành sắp xếp lại (Re-ranking) toàn bộ danh sách ứng viên theo final\_score giảm dần, cắt lấy 5 kết quả cao nhất (Top-5) và ánh xạ ngược về ID thực tế để trả về cho người dùng.

## <a name="_heading=h.rzlj4hntx9k0"></a>**6.3. TỐI ƯU HÓA KHÔNG GIAN TÌM KIẾM QUY MÔ  LỚN VỚI FAISS**
Khi thực nghiệm trên toàn bộ 34.250 ảnh, việc tính toán độ tương đồng theo cách brute-force giữa từng ảnh truy vấn và toàn bộ gallery có thể gây tốn thời gian. Vì vậy, nhóm sử dụng FAISS để tối ưu tốc độ truy xuất trên không gian vector đặc trưng.

- **Cấu hình chỉ mục:** Nhóm sử dụng faiss.IndexFlatIP. Khi các vector đã được chuẩn hóa L2, phép Inner Product trong faiss.IndexFlatIP tương đương với Cosine Similarity.
- **Quy trình truy xuất:** Toàn bộ vector đặc trưng sau khi chuẩn hóa được đưa vào chỉ mục FAISS. Với mỗi ảnh truy vấn, hệ thống tìm kiếm các vector có độ tương đồng cao nhất và trả về danh sách Top-K kết quả.
- **Hậu xử lý:** Hệ thống loại bỏ chính ảnh query khỏi danh sách kết quả trước khi tính các chỉ số Precision@K, Recall@K và mAP@K.
## <a name="_heading=h.h6k1zdcnr3z0"></a>**6.4. CHIẾN LƯỢC TỐI ƯU HÓA SIÊU THAM SỐ VÀ CƠ CHẾ TÁI XẾP HẠNG (RE-RANKING)**
Sau khi đánh giá các phương pháp cơ sở, nhóm nhận thấy việc ghép nối trực tiếp (concatenate) các vector đặc trưng hoặc thiết lập trọng số theo cảm tính (heuristic) dễ dẫn đến nhiễu loạn không gian vector và thiên lệch (bias). Để tối ưu hóa năng lực truy xuất, nhóm chuyển sang hướng tiếp cận **Hòa trộn Đặc trưng (Feature Fusion)** ở Giai đoạn 1 và **Hòa trộn Điểm số (Score Fusion)** ở Giai đoạn 2. Toàn bộ siêu tham số đều được tìm kiếm dạng lưới (Grid Search) một cách minh bạch trên tập Validation.
### <a name="_heading=h.uft282peuryx"></a>6.4.1. Tối ưu trọng số hòa trộn đa phương thức (alpha) tại Giai đoạn 1
Tại Giai đoạn 1, hệ thống sử dụng MobileCLIP để trích xuất Vector Hình ảnh (q\_img) và Vector Văn bản (q\_txt). Để tạo ra vector truy vấn tổng hợp (q\_fused), nhóm áp dụng công thức hòa trộn tuyến tính:

![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.008.png)

Trong đó, alpha là hệ số điều chỉnh mức độ đóng góp giữa đặc trưng hình ảnh và văn bản.

**Phương pháp dò tìm alpha:**

Nhóm tiến hành thực nghiệm Grid Search trên tập Validation (6.850 ảnh), quét giá trị alpha từ 0.1 đến 1.0 để chọn cấu hình cho mAP@5 cao nhất:

- Khi alpha = 0.1 → 0.4: Vectơ văn bản chiếm ưu thế (60-90%). Kết quả truy xuất bị nhiễu nặng do đặc thù tiêu đề Shopee chứa nhiều từ khóa rác (spam keywords), khiến mAP@5 ở mức thấp.
- Khi alpha = 0.9: Vectơ hình ảnh chiếm 90%. Mô hình gần như bỏ qua các từ khóa ngữ cảnh quan trọng (như mã model, thương hiệu), làm suy giảm độ chính xác.
- **Kết luận:** Quỹ đạo của mAP@5 đạt giá trị tốt nhất trong thực nghiệm tại giá trị **alpha = 0.5** (Hình ảnh đóng góp 50%, Văn bản đóng góp 50%). Cấu hình này cung cấp điểm cân bằng hoàn hảo.
### <a name="_heading=h.9x23zbfmf0bb"></a>6.4.2. Cơ chế Tái xếp hạng ứng viên (Candidate Re-ranking) bằng DINOv2
Nếu sử dụng mạng thị giác sâu DINOv2 để quét toàn bộ 34.250 ảnh cho mỗi lượt truy vấn, chi phí tính toán và bộ nhớ VRAM sẽ bị quá tải (OOM). Do đó, nhóm thiết kế kiến trúc Truy xuất 2 giai đoạn (Two-stage Retrieval Pipeline) với cơ chế Score Fusion như sau:

- **Bước 1: Lọc ứng viên thô (Candidate Generation)**\
  Thay vì tìm ngay Top-5, hệ thống sử dụng giải thuật FAISS để quét nhanh Vector Đa phương thức (MobileCLIP) trên toàn bộ Gallery. Hệ thống trích xuất ra một danh sách ứng viên rộng hơn (Top-retrieval\_k, ví dụ K=100). Bước này tốn rất ít tài nguyên, chạy trong thời gian thực và đảm bảo tỷ lệ bao phủ (Recall@100) cực kỳ cao. Điểm số tương đồng tại bước này được lưu lại thành clip\_score.
- **Bước 2: Trích xuất đặc trưng thị giác sâu (DINOv2 Re-scoring)**\
  Hệ thống chỉ duyệt qua danh sách ứng viên thu được từ Bước 1. Mạng DINOv2\_vitb14 được kích hoạt để soi chiếu các chi tiết thị giác vi mô (màu sắc, họa tiết, góc cạnh) của ảnh truy vấn và ảnh ứng viên. Điểm tương đồng mới được tính bằng Cosine Similarity và gọi là dino\_score.
- **Bước 3: Hòa trộn điểm số và Xếp hạng lại (Score Fusion & Re-ranking)**\
  Để không làm mất đi thông tin ngữ nghĩa (Title) đã phân tích ở Giai đoạn 1, hệ thống không dùng DINOv2 để thay thế hoàn toàn, mà tiến hành hòa trộn 2 luồng điểm số theo công thức:\
  ![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.009.png)\
  Trọng số beta tiếp tục được nội suy từ tập Validation. Sau khi có final\_score, danh sách ứng viên được sắp xếp lại theo thứ tự giảm dần. Khâu này đóng vai trò như một "bộ lọc tinh", đánh tụt hạng các sản phẩm có màu nền giống nhau nhưng sai khác về chi tiết cấu tạo.
- **Bước 4: Trích xuất kết quả (Top-K Extraction)**\
  Hệ thống cắt lấy đúng 5 kết quả đứng đầu từ danh sách đã được sắp xếp lại. Đây chính là danh sách Top-5 cuối cùng được ánh xạ ngược về ID gốc để đưa vào đánh giá các chỉ số mAP@5, Precision@1 và Recall@5 trên tập Test.

Kết quả thử nghiệm chứng minh kiến trúc MobileCLIP kết hợp DINOv2 Score Fusion đem lại sự cải thiện đột phá và ổn định hơn hẳn so với các cơ chế cộng điểm (Boosting) thô sơ
# <a name="_heading=h.vlgie2qds2ei"></a>**7. KẾT QUẢ THỰC NGHIỆM VÀ BẢNG SO SÁNH**


|**Phương pháp**|**Precision@1**|**Recall@5**|**mAP@5**|
| :-: | :-: | :-: | :-: |
|ResNet50 tuần 4|0\.6328|0\.5293|0\.5268|
|Baseline pHash tuần 4|0\.3840|0\.3095|0\.2509|
|Baseline TF-IDF tuần 4|0\.6929|0\.7336|0\.5735|
|CLIP image-to-image tuần 4|0\.6186|0\.5266|0\.5492|
|YOLO + SAHI + Crop Bounding Boxes + DINOv2 + TF\_IDF+ pHash + Candidate Union + Re-ranking|0\.7981|0\.738|0\.7738|
|Baseline EfficientNetB0 + MiniLM |0\.7548|0\.6796|0\.7107|
|Baseline MobileCLIP|0\.6772|0\.6266|0\.6307|
|MobileCLIP + DINOv2 + re-ranking + alpha search grid|0\.8064|0\.753|0\.7872|

` `Bảng 4: So sánh kết quả thực nghiệm giữa các phương pháp



![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.010.png)

Bảng 5: Biểu đồ so sánh kết quả thực nghiệm

![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.011.png)

Dựa trên biểu đồ đối sánh thời gian truy xuất (Latency), việc chuyển đổi từ phương pháp tính toán khoảng cách vét cạn truyền thống sang hệ thống lõi FAISS (cấu trúc IndexFlatIP) đã mang lại một bước nhảy vọt về mặt hiệu năng vận hành. Cụ thể:

- **Tốc độ xử lý vượt trội:** Thời gian thực thi trung bình cho một lượt truy vấn (Query) đã giảm mạnh từ mức 0.1656 giây (Brute-force) xuống chỉ còn 0.0307 giây (FAISS). Tốc độ này tương đương với mức cải thiện hiệu năng gấp **5.4 lần**.
- **Bản chất của sự tối ưu (Góc nhìn MLOps):** Về mặt toán học, cấu trúc IndexFlatIP thực chất vẫn là một dạng quét cạn (Exact Search - tính toán Tích vô hướng với toàn bộ dữ liệu). Tuy nhiên, nguyên nhân FAISS nhanh hơn Brute-force (thường viết bằng vòng lặp Python, NumPy hoặc PyTorch thô) là nhờ kiến trúc được viết bằng C++ tối ưu hóa ở tầng phần cứng thấp. FAISS tận dụng xuất sắc các thư viện tính toán ma trận chuyên dụng (BLAS) và khả năng quản lý băng thông bộ nhớ (Memory Bandwidth) hiệu quả hơn, đặc biệt là khi kết hợp cùng kiến trúc xử lý song song của GPU.
- **Ý nghĩa thực tiễn đối với hệ thống E-commerce:** Trong môi trường thương mại điện tử, việc duy trì độ trễ (latency) ở mức dưới 50 mili-giây (0.05s) là yếu tố bắt buộc để đảm bảo trải nghiệm người dùng theo thời gian thực (Real-time UX). Không chỉ đáp ứng tiêu chuẩn này, mức tăng tốc 5.4x của FAISS còn giúp hệ thống giải phóng đáng kể tài nguyên máy chủ. Điều này chứng minh rằng pipeline của nhóm hoàn toàn có đủ năng lực để mở rộng quy mô (Scale-up) dữ liệu từ 34.250 ảnh lên mức hàng triệu sản phẩm mà vẫn duy trì được tốc độ phản hồi ổn định.



![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.012.png)

Dữ liệu trong bài báo E-commerce Product Similarity Match Detection using Product Text and Images

Theo số liệu tham chiếu từ bài báo, việc nâng cấp từ mạng CNN cơ sở (ResNet-18) lên các kiến trúc học sâu phức tạp (Siamese ResNet-50) đã giúp chỉ số đối sánh cải thiện mạnh mẽ (CV score tăng từ 0.612 lên 0.722). Quy luật này cũng hoàn toàn tương đồng với định hướng tối ưu hóa trong đồ án của nhóm: Việc từ bỏ các kiến trúc đơn giản để chuyển sang ứng dụng sức mạnh của các mạng đa phương thức hiện đại (như MobileCLIP kết hợp DINOv2) là một bước đi mang tính tất yếu để bứt phá giới hạn hiệu năng của hệ thống 

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

![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.013.png)

DINOv2: Learning Robust Visual Features without Supervision
# <a name="_heading=h.xjm4lr56zvp6"></a>**8. PHÂN TÍCH LỖI CHUYÊN SÂU**
Bên cạnh việc báo cáo các chỉ số tổng hợp như Precision@1, Recall@5 và mAP@5, nhóm thực hiện phân tích lỗi để xác định những nhóm truy vấn mà phương pháp chính vẫn chưa xử lý tốt. Phần này được xây dựng trực tiếp từ notebook Tuan4\_QuocBao\_ErrorAnalysis.ipynb, sử dụng các file kết quả chi tiết của pipeline Strong Fusion và danh sách Top-K đã lưu sau thực nghiệm. Mục tiêu không chỉ là chỉ ra mô hình sai ở đâu, mà còn giải thích vì sao các lỗi đó xuất hiện trong bối cảnh dữ liệu Shopee có nhiều nhiễu nền, nhiễu văn bản và phân phối nhãn dạng long-tail.
## <a name="_heading=h.dsmhwp1lpb8m"></a>**8.1. Quy trình trích xuất mẫu lỗi**
Nhóm xác định lỗi nghiêm trọng bằng điều kiện AP@5 = 0. Với điều kiện này, một ảnh truy vấn được xem là thất bại hoàn toàn trong phạm vi Top-5 vì hệ thống không trả về ảnh nào cùng label\_group với ảnh truy vấn. Sau khi lọc các truy vấn có AP@5 = 0, notebook tiếp tục lấy kết quả Top-1 đầu tiên khác chính ảnh truy vấn để so sánh trực quan giữa ảnh query và ảnh trả về sai.

Quy trình phân tích gồm bốn bước: đọc file kết quả chi tiết STRONG\_best\_detail\_\*.csv, đọc ma trận STRONG\_best\_top\_indices\_\*.npy, ánh xạ chỉ số ảnh về bảng candidate\_df\_strong\_yolo\_crop\_fusion.csv, sau đó xuất bảng error\_analysis.csv gồm 10 mẫu lỗi đại diện. Cách làm này giúp phần phân tích lỗi bám sát kết quả thực nghiệm thực tế thay vì chỉ nhận xét cảm tính từ một vài ảnh minh họa rời rạc.

|**Nội dung thống kê**|**Giá trị**|**Ý nghĩa**|
| :-: | :-: | :-: |
|Số truy vấn đánh giá|34\.250|Mỗi ảnh trong tập dữ liệu được dùng làm một truy vấn.|
|Số cột trong detail\_df|13|Lưu method, query\_idx, label\_group và các metric AP/Precision/Recall.|
|Kích thước top\_indices|34\.250 x 10|Mỗi truy vấn có danh sách 10 ứng viên đầu tiên để phân tích Top-K.|
|Số truy vấn AP@5 = 0|3\.053|Các truy vấn không có ảnh đúng trong Top-5.|
|Tỷ lệ lỗi hoàn toàn|8,91%|Cho thấy vẫn tồn tại một nhóm truy vấn khó dù mAP@5 tổng thể cao.|
|Số mẫu lỗi đưa vào phân tích|10|Chọn cân bằng từ các nhóm lỗi chính để phân tích định tính.|


## <a name="_heading=h.lyaierkk68jz"></a>**8.2. Cấu trúc bảng phân tích lỗi**
File error\_analysis.csv được dùng làm minh chứng cho quá trình phân tích lỗi. Mỗi dòng tương ứng với một truy vấn bị sai hoàn toàn trong Top-5, kèm theo ảnh Top-1 mà mô hình trả về sai và nhóm nguyên nhân được gán cho lỗi đó.

|**Trường dữ liệu**|**Ý nghĩa**|
| :-: | :-: |
|query\_image|Tên file ảnh truy vấn bị truy xuất sai hoàn toàn.|
|top1\_result|Ảnh đứng đầu danh sách kết quả nhưng không cùng label\_group với ảnh truy vấn.|
|label\_query|Nhãn đúng của ảnh truy vấn.|
|label\_top1|Nhãn của ảnh Top-1 bị trả về sai.|
|error\_type|Nhóm lỗi được phân loại, ví dụ lỗi do nền/màu, góc chụp hoặc chữ quảng cáo.|
|lý do sai|Giải thích ngắn gọn nguyên nhân khiến mô hình nhầm lẫn.|


## <a name="_heading=h.uqg78t59gdnu"></a>**8.3. Kết quả phân loại 10 mẫu lỗi đại diện**
Từ 3.053 truy vấn có AP@5 = 0, nhóm chọn 10 mẫu đại diện theo hướng cân bằng giữa các nhóm lỗi phổ biến. Bảng dưới đây không nhằm mô tả phân phối lỗi của toàn bộ tập dữ liệu, mà dùng để minh họa các tình huống sai điển hình cần phân tích sâu.

|**STT**|**Query image**|**Top-1 sai**|**Label query**|**Label Top-1**|**Nhóm lỗi**|
| :-: | :-: | :-: | :-: | :-: | :-: |
|1|795c6f8f...37f0.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.014.jpeg)|f3c5cc20...1425.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.015.jpeg)|3637220226|2240027280|Màu sắc/nền giống nhau|
|2|9720558f...b8fc.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.016.jpeg)|5f0872e1...91c4.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.017.jpeg)|1445188681|4117627196|Màu sắc/nền giống nhau|
|3|e1a5f03b...3c99.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.018.jpeg)|0e9d686b...8e21.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.019.jpeg)|2699450457|514101767|Màu sắc/nền giống nhau|
|4|ed44a3b1...3af.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.020.jpeg)|e185854c...6c83.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.021.jpeg)|2304809467|2608223592|Màu sắc/nền giống nhau|
|5|685c22d9...809b.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.022.jpeg)|1d1ac481...0d2.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.023.jpeg)|1706249589|3097344893|Góc chụp khác nhau|
|6|c99e19b4...2d52.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.024.jpeg)|6af85140...18e.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.025.jpeg)|3947628716|2221700681|Góc chụp khác nhau|
|7|fb2d9463...77f.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.026.jpeg)|16f81057...2c5.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.027.jpeg)|1060961612|4032101535|Góc chụp khác nhau|
|8|a92eb87e...bfd.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.028.jpeg)|4b8affda...7c6.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.029.jpeg)|169470278|38185708|Góc chụp khác nhau|
|9|39e80002...94f.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.030.jpeg)|34e78d71...6b.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.031.jpeg)|2342771184|3135057640|Chữ quảng cáo che sản phẩm|
|10|64ac780d...323f.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.032.jpeg)|5c2d35f3...e89.jpg![](Aspose.Words.ee11b133-30a9-4120-a41f-19bec3f991be.033.jpeg)|1680934077|871931738|Chữ quảng cáo che sản phẩm|



Trong 10 mẫu đại diện, nhóm ghi nhận 4 trường hợp thuộc nhóm lỗi do màu sắc hoặc nền giống nhau, 4 trường hợp do khác biệt góc chụp và 2 trường hợp do chữ quảng cáo hoặc chi tiết chữ trên ảnh làm nhiễu. Vì đây là tập mẫu được chọn cân bằng để phân tích định tính, tỷ lệ 4-4-2 không được xem là tỷ lệ lỗi toàn cục của toàn bộ 34.250 truy vấn.
## <a name="_heading=h.50aasyu9iosw"></a>**8.4. Phân tích nguyên nhân lỗi**
**Nhóm lỗi 1 - Màu sắc hoặc nền giống nhau:** Các ảnh sản phẩm thương mại điện tử thường được chụp trên nền trắng, nền trơn hoặc có bố cục đặt sản phẩm gần giống nhau. Khi hai ảnh có màu chủ đạo, ánh sáng hoặc vùng nền tương đồng, vector đặc trưng toàn cục có thể bị kéo về gần nhau dù sản phẩm thực tế khác label\_group. Đây là hạn chế thường gặp của các mô hình dựa nhiều vào đặc trưng tổng thể, đặc biệt khi đối tượng chính nhỏ hoặc bị nhiễu bởi nền.

**Nhóm lỗi 2 - Khác biệt góc chụp, tỷ lệ và bố cục:** Một sản phẩm có thể được chụp ở nhiều góc, nhiều khoảng cách hoặc trong các điều kiện crop khác nhau. Khi ảnh query và ảnh liên quan khác mạnh về phối cảnh, mô hình có xu hướng xem chúng là hai đối tượng khác nhau. Ngược lại, một sản phẩm khác nhưng có cùng góc chụp và bố cục lại có thể được xếp hạng cao hơn. Điều này cho thấy pipeline vẫn cần cơ chế biểu diễn ổn định hơn trước các biến đổi hình học như góc nhìn, scale và vị trí sản phẩm trong ảnh.

**Nhóm lỗi 3 - Chữ quảng cáo và nhiễu văn bản trên ảnh:** Một số ảnh Shopee chứa banner, nhãn dán, giá khuyến mãi hoặc chữ quảng cáo chiếm diện tích lớn. Các vùng chữ này làm thay đổi đặc trưng thị giác của ảnh và có thể khiến mô hình ưu tiên thông tin không thuộc bản chất sản phẩm. Ngoài ra, tiêu đề sản phẩm cũng thường chứa từ khóa quảng cáo hoặc từ khóa được nhồi để tăng khả năng tìm kiếm, làm nhiễu nhánh văn bản của mô hình đa phương thức.

**Tác động của dữ liệu long-tail:** Phần lớn label\_group trong bộ dữ liệu chỉ có rất ít ảnh liên quan. Với các nhóm chỉ có 2 ảnh, sau khi loại bỏ chính ảnh truy vấn, hệ thống thường chỉ còn 1 ảnh đúng trong gallery. Khi ảnh đúng duy nhất này không nằm trong Top-5, AP@5 lập tức bằng 0. Do đó, các truy vấn thuộc nhóm ít mẫu làm cho chỉ số truy xuất rất nhạy với sai lệch thứ hạng.
## <a name="_heading=h.af40gebi1f3a"></a>**8.5. Độ tin cậy của quy trình phân loại lỗi và hướng cải tiến**
Việc phân loại nhóm lỗi ban đầu được thực hiện bán tự động bằng các heuristic đơn giản trên ảnh, bao gồm khoảng cách màu trung bình, tỷ lệ nền trắng ở vùng biên, mật độ cạnh và một số tín hiệu liên quan đến chữ quảng cáo. Sau đó, nhóm kiểm tra trực quan các cặp ảnh query/Top-1 để xác nhận lại nguyên nhân sai. Vì vậy, kết quả phân tích lỗi có vai trò hỗ trợ định tính, giúp nhận diện xu hướng lỗi phổ biến của mô hình, không được xem là nhãn lỗi tuyệt đối cho toàn bộ tập dữ liệu.

Từ các lỗi quan sát được, nhóm xác định một số hướng cải tiến hợp lý cho giai đoạn tiếp theo: bổ sung bước crop hoặc segmentation để giảm ảnh hưởng của phông nền; sử dụng OCR hoặc bộ lọc vùng chữ để hạn chế nhiễu quảng cáo; thêm đặc trưng màu cục bộ như HSV histogram hoặc texture descriptor để phân biệt các sản phẩm cùng kiểu dáng nhưng khác màu; thử nghiệm Query Expansion hoặc K-reciprocal Re-ranking để cải thiện truy xuất trên các nhóm có ít ảnh liên quan. Các hướng này bám trực tiếp vào lỗi thực nghiệm thay vì chỉ thay đổi mô hình theo cảm tính. 
# <a name="_heading=h.ynorrnpdrefv"></a>**9. DEMO HOẶC SẢN PHẨM THỬ NGHIỆM**
Để minh chứng khả năng hoạt động thực tế của hệ thống truy xuất hình ảnh sản phẩm tương đồng, nhóm đã xây dựng một notebook demo với tên Demo.ipynb. Demo này cho phép người dùng nhập vào một ảnh truy vấn bất kỳ, sau đó hệ thống sẽ trả về danh sách Top-5 ảnh sản phẩm tương đồng nhất dựa trên kết quả truy xuất đã được tạo bởi phương pháp chính. 
## <a name="_heading=h.27p8pvrw879p"></a>**9.1. Mục tiêu của demo**
Mục tiêu của demo là mô phỏng quy trình sử dụng hệ thống Visual Search ở mức đơn giản, trực quan và dễ kiểm chứng. Người dùng chỉ cần cung cấp một ảnh sản phẩm đầu vào, hệ thống sẽ tìm kiếm trong tập dữ liệu và hiển thị 5 ảnh có độ tương đồng cao nhất. Kết quả demo giúp nhóm kiểm tra trực tiếp khả năng truy xuất của mô hình, đồng thời hỗ trợ quan sát các trường hợp đúng và sai trong danh sách Top-K.

**Cấu trúc sử dụng chính của demo:**

query\_path = 'sample\_query.jpg'

top5 = search(query\_path, index, candidate\_df, IMAGE\_DIR, k=5)

show\_top5(top5)

Trong đó, query\_path là đường dẫn đến ảnh truy vấn, search() là hàm thực hiện truy xuất ảnh tương đồng, còn show\_top5() là hàm hiển thị trực quan ảnh truy vấn cùng 5 kết quả tìm được.
## <a name="_heading=h.nrd5pu59018z"></a>**9.2. Dữ liệu và file đầu vào của demo**
Notebook demo sử dụng các file kết quả đã được tạo ra từ pipeline chính của nhóm, bao gồm:

|**Thành phần**|**Vai trò**|
| :-: | :-: |
|candidate\_df\_strong\_yolo\_crop\_fusion.csv|Lưu thông tin ảnh, label\_group, tên file ảnh và các thông tin phục vụ truy xuất.|
|STRONG\_best\_top\_indices\_\*.npy|Lưu danh sách chỉ số Top-K đã được tính sẵn bởi phương pháp chính.|
|train\_images|Thư mục chứa toàn bộ ảnh sản phẩm trong tập dữ liệu.|
|sample\_query.jpg|Ảnh truy vấn mẫu do người dùng cung cấp.|
|results/demo\_top5\_result.csv|File lưu kết quả Top-5 sau khi chạy demo.|

Notebook được thiết kế để tự động tìm kiếm các file cần thiết trong thư mục dự án. Trong trường hợp không tìm được, người dùng có thể cấu hình thủ công đường dẫn thông qua các biến MANUAL\_TOP\_NPY, MANUAL\_CANDIDATE\_CSV và MANUAL\_IMAGE\_DIR.
## <a name="_heading=h.exxgmdhdm2nw"></a>**9.3. Quy trình hoạt động của demo**
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
## <a name="_heading=h.ukzp824u7i6m"></a>**9.4. Kết quả đầu ra của demo**
Kết quả đầu ra của demo gồm ba phần chính:

|**Đầu ra**|**Mô tả**|
| :-: | :-: |
|Hình ảnh trực quan|Hiển thị ảnh truy vấn và 5 ảnh sản phẩm tương đồng nhất.|
|Bảng Top-5|Hiển thị thứ hạng, phương pháp truy xuất, tên ảnh kết quả, nhãn sản phẩm và trạng thái đúng/sai.|
|File CSV|Lưu kết quả truy xuất vào results/demo\_top5\_result.csv.|

Thông qua demo, nhóm có thể quan sát trực tiếp hiệu quả của hệ thống thay vì chỉ đánh giá bằng các chỉ số định lượng như Precision@1, Recall@5 và mAP@5. Điều này giúp việc phân tích lỗi trở nên rõ ràng hơn, đặc biệt trong các trường hợp mô hình trả về ảnh có màu sắc hoặc phông nền giống nhau nhưng khác nhãn sản phẩm.
## <a name="_heading=h.1byjm1hck9vs"></a>**9.5. Nhận xét về sản phẩm thử nghiệm**
Demo đã đáp ứng được yêu cầu cơ bản của một hệ thống Visual Search thử nghiệm: nhận ảnh đầu vào, truy xuất ảnh tương đồng và hiển thị kết quả Top-5 một cách trực quan. Notebook có khả năng tự động tìm các file kết quả đã sinh ra từ pipeline chính, giúp giảm phụ thuộc vào đường dẫn cố định trên từng máy. Ngoài ra, cơ chế lưu kết quả ra file CSV giúp nhóm dễ dàng kiểm tra, so sánh và đưa kết quả vào báo cáo.

Tuy nhiên, demo hiện vẫn ở mức notebook thử nghiệm, chưa được đóng gói thành giao diện người dùng hoàn chỉnh. Trong các tuần tiếp theo, nhóm có thể phát triển thêm giao diện bằng Streamlit để người dùng tải ảnh trực tiếp, xem kết quả Top-5 trên trình duyệt và thao tác dễ dàng hơn.
# <a name="_heading=h.wl3degyuh5z7"></a>**10. PHÂN CÔNG, MINH CHỨNG CÁ NHÂN VÀ KHAI BÁO SỬ DỤNG AI**
Bảng Ghi chú AI hỗ trợ

|**Phần**|**AI/model hỗ trợ** |**AI hỗ trợ như thế nào**|**Người kiểm tra & giải thích**|
| :-: | :-: | :-: | :-: |
|<p>2 Mô hình Baseline </p><p>Baseline1\_MobileCLIP.ipynb</p><p>Baseline2\_EfficientNetB0\_MiniLM.ipynb</p>|Claude |AI hỗ trợ gợi ý triển khai 2 baseline, thay đổi đường dẫn và chạy trên gg colab|Mã Gia Vỹ |
|Tuan4\_QuocBao\_ErrorAnalysis.ipynb|Claude|AI hỗ trợ gợi ý triển khai|Lê Quốc Bảo|
|Tuan4\_KhanhHung\_MetricUnify.ipynb|Gemini |AI hỗ trợ gợi ý triển khai|Nguyễn Khánh Hưng|
|Tuan4\_GiaVy\_Pipeline.ipynb|Claude|AI hỗ trợ gợi ý pipeline 2 giai đoạn MobileCLIP , re-ranking bằng DINOv2 và candidate re-ranking. Nhóm mới thử nghiệm bước đầu, ghi nhận kết quả khả quan và dự kiến tiếp tục nghiên cứu trong Tuần 5. |Mã Gia Vỹ|
|Tuan4\_StrongFusion\_YOLOCrop\_DINOv2\_TFIDF\_pHash.ipynb|Claude|AI hỗ trợ gợi ý triển khai|Lê Quốc Bảo|


# <a name="_heading=h.h8ep106zfjws"></a>**11. KẾ HOẠCH TUẦN 5**
Bảng 5: Kế hoạch tuần 4

|Nội dung cải tiến|Phương pháp / Thuật toán đề xuất|Mục tiêu cốt lõi|
| :-: | :-: | :-: |
|1\. Triển khai luồng Nhận diện & Cắt vùng (Bám sát định hướng Giảng viên) |Tích hợp mạng YOLO kết hợp kỹ thuật phân mảnh SAHI để phát hiện vật thể. Cắt (crop) Bounding Box của sản phẩm chính trước khi đưa vào mô hình DINOv2. |Khử triệt để phông nền phức tạp, watermark và chữ quảng cáo rác. Buộc hệ thống đánh giá (Re-ranking) phải tập trung 100% vào hình dáng sản phẩm cốt lõi, triệt tiêu lỗi nhận diện nhầm do trùng màu nền. |
|2\. Màng lọc siêu tốc với Mã băm Thị giác (Perceptual Hash)|Tích hợp thuật toán pHash làm màng lọc sớm (Early-pass filter). Nếu khoảng cách Hamming giữa truy vấn và ảnh trong Gallery approx 0, cộng trực tiếp điểm thưởng (pHash Boosting).|Xử lý chớp nhoáng các trường hợp ảnh copy-paste, ảnh chỉ đổi kích thước/độ sáng. Giúp hệ thống đẩy ngay kết quả trùng lặp tuyệt đối lên Top 1 mà không cần tốn tài nguyên chạy Deep Learning.|
|3\. Kiến trúc Truy xuất Lai (Hybrid Multi-Stage Pipeline)|Xây dựng luồng xử lý phân tầng: (1) pHash (bắt ảnh trùng lặp) → (2) MobileCLIP (lọc top ứng viên từ ảnh gốc) → (3) YOLO Crop (cắt vật thể ứng viên) → (4) DINOv2 (tính điểm chi tiết vật thể cắt).|Hợp nhất ưu điểm của mọi phương pháp: Tốc độ của pHash, Khả năng hiểu ngữ cảnh của MobileCLIP, và Năng lực soi chi tiết không lẫn tạp âm của YOLO + DINOv2.|
|4\. Tối ưu hóa Triển khai (MLOps & Hardware Optimization)|Tối ưu hóa suy luận (Inference Optimization) bằng cách chuyển đổi trọng số YOLO sang định dạng ONNX hoặc dùng kỹ thuật Batch Inference để xử lý hàng loạt.|Giải quyết triệt để nút thắt cổ chai về tràn bộ nhớ (OOM) và Session Timeout khi chạy YOLO/SAHI trên 34.250 ảnh với phần cứng giới hạn (Google Colab T4).|
|5\. Kiến trúc Hệ thống (Client-Server)|Đóng gói hệ thống độc lập: Backend xử lý API bằng FastAPI/Flask. Frontend sử dụng Streamlit/Gradio hoặc web-based stack (HTML/JS). Quy hoạch dự án chuẩn kỹ sư phần mềm (phân luồng rõ thư mục src/, docs/, tools/).|Đảm bảo hệ thống vận hành trơn tru, dễ dàng bảo trì. Giao diện có tính đáp ứng cao (Responsive), hiển thị mượt mà trên cả màn hình máy tính lẫn thiết bị di động.|
|6\. Đầu vào Đa phương thức (Multimodal Input)|Cung cấp khu vực Upload cho phép kéo thả ảnh (Drag & Drop). Nâng cao: Tích hợp thanh tìm kiếm văn bản (Search bar) bổ trợ ngay bên dưới.|Mô phỏng chính xác hành vi tìm kiếm trên Shopee. Người dùng có thể linh hoạt kết hợp ảnh và từ khóa (VD: Upload ảnh áo thun + gõ thêm "màu đen") để thu hẹp kết quả.|
|7\. Trực quan hóa AI (AI Explainability)|Trong thời gian chờ (Loading), hệ thống gọi kết quả từ mạng YOLO để vẽ trực tiếp khung nhận diện (Bounding Box) đè lên bức ảnh gốc người dùng vừa tải lên.|Đây là điểm nhấn công nghệ: Giúp người dùng "nhìn thấy" AI đang khoanh vùng vào đúng chi tiết nào của bức ảnh, làm tăng độ tin cậy và sự thích thú khi trải nghiệm.|
|8\. Hiển thị Kết quả (Result Grid)|Danh sách trả về hiển thị dạng lưới (Grid Layout). Mỗi thẻ sản phẩm (Product Card) đính kèm đầy đủ: Hình ảnh, Nhãn danh mục (Label) và Điểm tương đồng (Similarity Score %).|Giúp người dùng dễ dàng đối chiếu sản phẩm bằng mắt thường. Các kết quả có điểm thưởng tuyệt đối từ pHash sẽ được gắn nhãn "Exact Match" nổi bật ở vị trí Top-1.|

# <a name="_heading=h.189v3ox24ggx"></a>**12. TÀI LIỆU THAM KHẢO**
[1] W. McKinney, "Data structures for statistical computing in Python," in *Proceedings of the 9th Python in Science Conference*, 2010, pp. 51–56. [Online]. 

[2] A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019. [Online]. 

[3] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011. [Online]. 

[4] J. D. Hunter, "Matplotlib: A 2D graphics environment," *Computing in Science & Engineering*, vol. 9, no. 3, pp. 90–95, 2007. [Online]. 

[5] M. Zhu, "Recall, Precision and Average Precision," University of Waterloo, 2004. [Online]. 

[6] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[7] J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," IEEE Transactions on Big Data, 2019.

[8] Kaggle, "Shopee - Price Match Guarantee," Kaggle Competition Dataset.

[9] A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," in Proceedings of the 38th International Conference on Machine Learning (ICML), 2021.

[10] M. Kaya and H. Ş. Bilge, "Deep Metric Learning: A Survey," Symmetry, vol. 11, no. 9, p. 1066, 2019.

[11] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019

[12] Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision". arXiv preprint arXiv:2304.07193.

- DINOv2 là một phương pháp học tự giám sát cho Vision Transformer, tạo ra các biểu diễn hình ảnh mạnh mẽ, vượt trội trong nhiều tác vụ thị giác mà không cần tinh chỉnh. Phiên bản Small (DINOv2\_vits14) được dùng trong pipeline của bạn. 

[13] Vasu, P. K. A., et al. (2024). "MobileCLIP: Fast Multimodal Learning for Mobile Devices". CVPR 2024 / arXiv preprint.

- Bài báo giới thiệu dòng mô hình MobileCLIP, được thiết kế để đạt hiệu suất cao trên thiết bị di động. MobileCLIP‑S1 (phiên bản bạn đang dùng) đạt độ chính xác zero‑shot tương đương CLIP ViT‑B/16 nhưng nhanh hơn gấp 2.3 lần và nhỏ hơn đáng kể. 

[14] Jochko, M., et al. (2023). "YOLOv8: A Comprehensive Review of the State-of-the-Art Object Detector". Ultralytics.

- YOLOv8 là phiên bản mới nhất trong dòng YOLO, được thiết kế để cân bằng giữa tốc độ và độ chính xác. Nó hỗ trợ nhiều tác vụ như phát hiện, phân đoạn, và phân loại, phù hợp cho bài toán phát hiện sản phẩm trong thương mại điện tử. 

[15] Akyon, F. C., Altinuc, S. O., & Temizel, A. (2022). "SAHI: Slicing Aided Hyper Inference for Small Object Detection". 2022 IEEE International Conference on Image Processing (ICIP).

- SAHI là một framework giúp cải thiện khả năng phát hiện vật thể nhỏ bằng cách cắt ảnh thành các lát (slices) và chạy inference trên từng lát, sau đó hợp nhất kết quả. Phương pháp này đặc biệt hữu ích cho ảnh có độ phân giải cao hoặc ảnh có nhiều sản phẩm nhỏ. 

[16] M. Jain, J. C. van Gemert, C. G. M. Snoek (University of Amsterdam) . "Efficient Large-Scale Multi-Modal Late Fusion for Video Retrieval". [7] Zhong, Z., Zheng, L., Cao, D., & Li, S. (2017). "Re-ranking Person Re-identification with k-reciprocal Encoding". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)./

- Đề xuất phương pháp re‑ranking dựa trên mã hóa k‑reciprocal, tính lại khoảng cách giữa ảnh query và gallery bằng cách khai thác thông tin từ các láng giềng chung. Phương pháp này cải thiện đáng kể mAP và CMC. 



37
