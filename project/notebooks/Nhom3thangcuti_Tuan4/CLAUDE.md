# CLAUDE.md – Hướng dẫn hoàn thành nhiệm vụ Tuần 4 (Mã Gia Vỹ)

## 🎯 Mục tiêu cá nhân

- **Tăng mAP@5 của phương pháp chính từ 0.7635 lên ≥ 0.80** trên tập Test 80% (≈27,400 ảnh).
- Thay thế ResNet50 bằng **SigLIP** (backbone chính) và **DINOv3** (dùng riêng cho rerank hoặc late fusion).
- Giữ lại toàn bộ kỹ thuật tăng cường: **pHash boost**, **weighted fusion (TF‑IDF + visual)**, **reranking**.
- Tuân thủ đúng giao thức đánh giá mới: **Gallery 34,250**, **Validation 20%**, **Test 80%** (chỉ chạy 1 lần).

## 🧠 Lựa chọn mô hình & chiến lược tổng thể

| Thành phần                  | Mô hình/Tên                                   | Lý do                                                                                          |
| --------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Backbone thị giác chính** | `google/siglip-so400m-patch14-384` (SigLIP 2) | Đa phương thức, hiểu sâu ngữ nghĩa ảnh - văn bản, mAP truy xuất cao nhất hiện nay.             |
| **Backbone phụ (rerank)**   | `dinov2_vitb14` (DINOv3)                      | Xuất sắc về chi tiết thị giác thuần túy, giúp sắp xếp lại chính xác hơn ở bước 2.              |
| **Fusion strategy**         | Weighted sum (late fusion)                    | Đơn giản, dễ grid search, đã chứng minh hiệu quả. Tỉ lệ tối ưu sẽ được tìm trên Validation.    |
| **pHash boost**             | Giữ nguyên ngưỡng ≤ 10                        | Tăng điểm cho các ảnh có mã băm gần giống – hiệu quả với sản phẩm cùng nhãn nhưng khác ảnh.    |
| **Reranking**               | DINOv3 + cosine similarity                    | Bước 2: lấy top‑K ứng viên từ SigLIP, tính lại điểm bằng DINOv3 (chỉ thị giác) để sắp xếp lại. |

> **Tại sao không dùng SigLIP cho cả 2 bước?**  
> SigLIP vốn đã rất mạnh về ngữ nghĩa toàn cục. Khi rerank bằng DINOv3, ta bổ sung khả năng phân biệt chi tiết (vải, họa tiết, hình dáng) mà SigLIP có thể bỏ qua. Đây là hướng “best of both worlds” đã được kiểm chứng (CoME‑VL, Mercari).

## 📁 Cấu trúc file đầu ra của bạn
