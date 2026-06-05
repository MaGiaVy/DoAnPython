# Visual Search Project - Error Analysis & Improvement Plan

## Thông tin hiện tại

### Baseline

| Model         | mAP@5  |
| ------------- | ------ |
| MobileCLIP S0 | 0.7651 |

### Current Best

| Model                       | mAP@5  |
| --------------------------- | ------ |
| MobileCLIP + DINOv2 + pHash | 0.7837 |

---

# Nguyên nhân chưa đạt mAP@5 = 0.80

Qua phân tích pipeline, hệ thống hiện tại không xuất hiện lỗi nghiêm trọng (critical bug).

Metric bị giới hạn chủ yếu bởi:

1. Score fusion chưa tối ưu
2. MobileCLIP S0 có năng lực biểu diễn thấp
3. pHash chưa được khai thác hiệu quả
4. Chưa tiền xử lý văn bản
5. Candidate retrieval đã bão hòa

---

# 1. Cải thiện Score Fusion

## Hiện tại

```python
combined = beta * dino_scores + (1-beta) * candidate_clip_scores
```

### Vấn đề

DINOv2 và MobileCLIP tạo ra phân phối cosine similarity khác nhau.

Ví dụ:

```text
DINO:
0.88
0.91
0.94

CLIP:
0.35
0.52
0.61
```

Khi cộng trực tiếp:

```python
combined = beta*dino + (1-beta)*clip
```

một mô hình sẽ chi phối mô hình còn lại.

---

## Khuyến nghị

Chuẩn hóa score trước khi fusion.

### Z-score normalization

```python
clip_scores = np.array(candidate_clip_scores)
dino_scores = np.array(dino_scores)

clip_scores = (
    clip_scores - clip_scores.mean()
) / (clip_scores.std() + 1e-8)

dino_scores = (
    dino_scores - dino_scores.mean()
) / (dino_scores.std() + 1e-8)

combined = beta*dino_scores + (1-beta)*clip_scores
```

---

### Softmax normalization

```python
from scipy.special import softmax

clip_scores = softmax(candidate_clip_scores)
dino_scores = softmax(dino_scores)

combined = beta*dino_scores + (1-beta)*clip_scores
```

---

### Kỳ vọng

```text
+1% ~ +3% mAP
```

---

# 2. Thay thế MobileCLIP S0

## Hiện tại

```python
MOBILECLIP_VARIANT = "mobileclip_s0"
```

---

## Đề xuất

### Option 1

```python
mobileclip_s1
```

### Option 2

```python
mobileclip_s2
```

---

## Dự kiến

| Model | mAP@5       |
| ----- | ----------- |
| S0    | 0.76 - 0.78 |
| S1    | 0.78 - 0.80 |
| S2    | 0.80 - 0.82 |

---

# 3. Tối ưu pHash

## Hiện tại

```python
HAMMING_THRESHOLD = 8
PHASH_BONUS = 0.10
```

```python
combined[i] += 0.10
```

---

## Vấn đề

Bonus cố định không phản ánh độ tương đồng thực tế.

---

## Đề xuất

### Bonus nhỏ hơn

```python
combined[i] += 0.02
```

---

### Hoặc scaling

```python
combined[i] *= 1.05
```

---

### Grid Search

```text
Threshold:
4
6
8
10

Bonus:
0.01
0.02
0.03
0.05
```

---

# 4. Tiền xử lý văn bản

## Hiện tại

Text được encode trực tiếp.

```python
clip_model.encode_text(title)
```

---

## Đề xuất

### Lowercase

```python
title = title.lower()
```

### Remove special characters

```python
import re

title = re.sub(r'[^a-z0-9\s]', ' ', title)
```

### Normalize spaces

```python
title = ' '.join(title.split())
```

---

## Ví dụ

### Trước

```text
ÁO THUN NAM ĐẸP FREESHIP !!!
```

### Sau

```text
ao thun nam dep freeship
```

---

## Kỳ vọng

```text
+1% ~ +2% mAP
```

---

# 5. Kiểm tra Feature Normalization

## Hiện tại

Một số nhánh MobileCLIP có normalize:

```python
feats = feats / feats.norm(
    dim=-1,
    keepdim=True
)
```

Tuy nhiên cần kiểm tra lại các nhánh fallback.

---

## Đề xuất

Sau mọi lần extract feature:

```python
feats = feats / (
    feats.norm(dim=-1, keepdim=True) + 1e-8
)
```

---

# 6. Candidate Retrieval đã bão hòa

Kết quả thử nghiệm:

| Top-K | mAP@5  |
| ----- | ------ |
| 50    | 0.7798 |
| 100   | 0.7801 |
| 150   | 0.7803 |
| 200   | 0.7803 |

---

## Kết luận

Tăng Top-K không còn cải thiện hiệu năng.

Bottleneck nằm ở:

```text
Feature Quality
```

không phải

```text
Candidate Quantity
```

---

# 7. Chia Dataset hợp lý hơn

## Hiện tại

```python
df_gallery = df.copy()
```

Gallery chứa toàn bộ dataset.

---

## Đề xuất

```text
Train Gallery
Validation Gallery
Validation Query
Test Gallery
Test Query
```

tách biệt hoàn toàn.

---

# 8. Thử Reciprocal Rank Fusion (RRF)

Thay vì:

```python
combined = beta*dino + (1-beta)*clip
```

---

Sử dụng:

```python
def rrf(rank1, rank2, k=60):
    return (
        1/(k + rank1)
        +
        1/(k + rank2)
    )
```

---

Ưu điểm:

- Không phụ thuộc scale score
- Ổn định hơn khi fusion nhiều mô hình

---

# Thứ tự ưu tiên triển khai

## Ưu tiên 1

Score Normalization + Fusion

Dự kiến:

```text
+0.01 ~ +0.03 mAP
```

---

## Ưu tiên 2

MobileCLIP S1 hoặc S2

Dự kiến:

```text
+0.02 ~ +0.04 mAP
```

---

## Ưu tiên 3

Text Cleaning

Dự kiến:

```text
+0.01 ~ +0.02 mAP
```

---

## Ưu tiên 4

RRF Fusion

Dự kiến:

```text
+0.005 ~ +0.015 mAP
```

---

# Mục tiêu

Hiện tại:

```text
mAP@5 = 0.7837
```

Sau các cải tiến:

```text
0.80 ~ 0.83
```

là mục tiêu khả thi mà không cần huấn luyện thêm mô hình mới.
