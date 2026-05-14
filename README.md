# 🔍 Hệ Thống Tìm Kiếm Ảnh Tương Tự Bằng AI

Dự án xây dựng hệ thống tìm kiếm ảnh tương tự sử dụng Deep Learning và FAISS nhằm hỗ trợ tìm kiếm các hình ảnh có nội dung gần giống nhau dựa trên đặc trưng ảnh.

---

# 📌 Yêu Cầu Hệ Thống

- Python 3.11 hoặc 3.12
- Git
- pip

Khuyến nghị sử dụng môi trường ảo (`venv`) để tránh lỗi thư viện.

---

# 📥 Clone Project

```bash
git clone https://github.com/MaGiaVy/DoAnPython.git
```

Di chuyển vào thư mục project:

```bash
cd DoAnPython/project
```

---

# 🛠️ Tạo Môi Trường Ảo

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 📦 Cài Đặt Thư Viện

```bash
pip install -r requirements.txt
```

---

# 📂 Tải Dataset

Tải dataset tại đây:

[Google Drive Dataset](https://drive.google.com/file/d/12La9dODAiew-yk1IC9jAJ47LxUkDZ6su/view)

Sau khi tải về:
- Giải nén dataset
- Đặt thư mục train_images vào:

```txt
project/data/raw
```

Ví dụ:

```txt
DoAnPython/
│
└── project/
    └── data/
        └── raw/
```

---

# ⚙️ Cấu Hình Đường Dẫn Dataset

Mở file:

```txt
project/notebooks/BTCT_TUAN2_33.ipynb
```

Tìm dòng tại Cell_2 :

```python
DATA_DIR = r'd:\Study\docs\Python\workspace\DoAnPython\project\data\raw'
```

Thay đổi thành đường dẫn phù hợp với máy của bạn.

Ví dụ:

```python
DATA_DIR = r'C:\Users\Admin\Documents\DoAnPython\project\data\raw'
```

---

# ▶️ Chạy Project

Mở notebook:

```txt
project/notebooks/BTCT_TUAN2_33.ipynb
```

Sau đó:
- Chọn đúng Python Interpreter / Kernel
- Run từng cell theo thứ tự

---

# 🚨 Lưu Ý

Nếu gặp lỗi thiếu thư viện:

```bash
pip install tên_thư_viện
```

Nếu gặp lỗi liên quan đến `pandas`, `torch`, `faiss`:
- Kiểm tra đúng phiên bản Python
- Khuyến nghị sử dụng Python 3.11 hoặc 3.12

---

# 🧠 Công Nghệ Sử Dụng

- Python
- PyTorch
- FAISS
- NumPy
- Pandas
- Streamlit
- Matplotlib
- Seaborn

---

# 👨‍💻 Thành Viên Thực Hiện

- Lê Quốc Bảo 
- Mã Gia Vỹ
- Nguyễn Khánh Hưng

---

# 📄 License

Dự án phục vụ mục đích học tập và nghiên cứu.
