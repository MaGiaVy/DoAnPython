# Tìm kiếm hình ảnh sản phẩm tương đồng

## Thành viên
- Lê Quốc Bảo – 3124410015
- Mã Gia Vỹ – 3124410414
- Nguyễn Khánh Hưng – 3124410129

## Hướng dẫn cài đặt và chạy

### Bước 1: Clone repo về máy
```bash
git clone https://github.com/MaGiaVy/DoAnPython
cd project
```

### Bước 2: Tải dataset Shopee
Vào link sau và tải về máy:  
https://www.kaggle.com/competitions/shopee-product-matching/data

Giải nén ra, bạn sẽ có các file:
```
train.csv
test.csv
train_images/
test_images/
```

### Bước 3: Sửa file .env
Mở file `.env` và sửa đường dẫn cho đúng máy bạn:
```
SHOPEE_DATA_DIR=C:\Users\TenBan\Downloads\shopee-product-matching
RESULTS_DIR=C:\Users\TenBan\project\results
```

### Bước 4: Cài thư viện
```bash
pip install -r requirements.txt
```

### Bước 5: Chạy notebook
Mở VS Code → mở file `notebooks/Nhom3_Tuan2_EDA.ipynb` → chọn kernel `.venv` → Run All

## Cấu trúc thư mục
```
project/
├── data/
│   └── raw/          ← để trống, dataset không push lên GitHub
├── notebooks/        ← notebook chính
├── results/          ← biểu đồ và kết quả (tự sinh ra khi chạy notebook)
├── .env.example      ← mẫu khai báo đường dẫn
├── .gitignore
└── requirements.txt
```

## Lưu ý
- File `.env` **không được push lên GitHub** (đã có trong `.gitignore`)
- Dataset Shopee **không được push lên GitHub** vì quá nặng (~10GB)
- Mỗi thành viên tự tạo file `.env` riêng trên máy mình
