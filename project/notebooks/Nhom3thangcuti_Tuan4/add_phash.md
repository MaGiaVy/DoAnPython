# YÊU CẦU: TÍCH HỢP PHASH BOOSTING VÀO PIPELINE MOBILECLIP + DINOV2

## 🎯 Mục tiêu

Thêm cơ chế pHash boosting vào giai đoạn 2 (score fusion) nhằm tăng độ chính xác cho các trường hợp query có ảnh gần như giống hệt trong gallery (cùng sản phẩm, khác nhẹ về ánh sáng/kích thước).

## 📦 Yêu cầu đầu vào

- `gallery_phashes`: list chứa pHash của toàn bộ 34.250 ảnh (theo thứ tự của df_gallery). Cache ra file `phashes_gallery.npy`.
- `query_phash`: pHash của ảnh query (tính trong lúc inference).

## 🔧 Vị trí sửa code

Chỉ sửa **hàm `search_two_stage`**, phần sau khi đã tính `combined = beta * dino_scores + (1-beta) * clip_scores`.

## 🧬 Tham số đề xuất (không cần grid search)

- `HAMMING_THRESHOLD = 5` (chỉ thưởng cho ảnh rất giống)
- `PHASH_BONUS = 0.03` (điểm thưởng nhẹ, đủ để phá thế hoà)

## 💻 Code cần thêm vào `search_two_stage`

```python
# Sau khi đã có combined = beta * dino_scores + (1-beta) * clip_scores
# và candidate_indices

q_phash = compute_phash(query_img_path)  # bạn cần có hàm compute_phash

for i, idx in enumerate(candidate_indices):
    hamming_dist = abs(q_phash - gallery_phashes[idx])
    if hamming_dist <= HAMMING_THRESHOLD:
        combined[i] += PHASH_BONUS

# Sau đó sắp xếp lại
new_order = np.argsort(combined)[::-1]
final_indices = [candidate_indices[i] for i in new_order[:final_k]]
final_scores  = [combined[i]             for i in new_order[:final_k]]
```
