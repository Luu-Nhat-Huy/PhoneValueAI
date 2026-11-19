# PhoneValue AI – Hệ thống định giá điện thoại cũ bằng Machine Learning

## 1. Giới thiệu

**PhoneValue AI** là ứng dụng web sử dụng Machine Learning để dự đoán giá bán lại của điện thoại đã qua sử dụng.

Hệ thống cho phép người dùng nhập thông tin điện thoại, nhận kết quả định giá, phân tích dữ liệu, theo dõi lịch sử định giá và tìm hiểu thông tin về các mô hình dự đoán.

Dự án được xây dựng bằng Python và Streamlit, sử dụng bộ dữ liệu gồm 1.000.000 bản ghi.

## 2. Mục tiêu dự án

- Xây dựng mô hình Machine Learning dự đoán giá bán lại điện thoại cũ.
- So sánh hiệu quả của nhiều thuật toán hồi quy.
- Phân tích các yếu tố ảnh hưởng đến giá trị điện thoại.
- Xây dựng ứng dụng web trực quan, dễ sử dụng.
- Lưu trữ và quản lý lịch sử định giá.
- Đánh giá hiệu quả mô hình thông qua MAE, RMSE và R².

## 3. Công nghệ sử dụng

| Công nghệ | Mục đích |
|---|---|
| Python 3.13.7 | Ngôn ngữ lập trình |
| Pandas, NumPy | Xử lý và phân tích dữ liệu |
| Scikit-learn | Xây dựng mô hình Machine Learning |
| Matplotlib, Seaborn | Trực quan hóa dữ liệu |
| Plotly | Xây dựng biểu đồ tương tác |
| Streamlit | Xây dựng giao diện web |
| Joblib | Lưu và tải mô hình |
| SQLite | Lưu lịch sử định giá |

## 4. Dữ liệu sử dụng

Dự án sử dụng bộ dữ liệu `used_phone_price_prediction_1M.csv.gz`.

Thông tin dữ liệu:

- Số lượng: 1.000.000 dòng.
- Số đặc trưng đầu vào: 27.
- Biến mục tiêu: `resale_price`.
- Định dạng: CSV nén bằng gzip.

Một số thông tin đầu vào:

- Thương hiệu và model điện thoại.
- Hệ điều hành.
- Giá mua ban đầu.
- Tuổi đời thiết bị.
- RAM và dung lượng lưu trữ.
- Điểm hiệu năng CPU.
- Điểm camera.
- Tình trạng pin.
- Tình trạng ngoại hình.
- Lịch sử sửa chữa.
- Tình trạng bảo hành.
- Nhu cầu thị trường.
- Loại người bán và khu vực thành phố.

### Đơn vị giá

Giá trong dataset được lưu theo đơn vị **nghìn VND**.

Ví dụ:

| Giá trị trong dataset | Giá trị quy đổi |
|---:|---:|
| 10.000 | 10.000.000 VND |
| 45.000 | 45.000.000 VND |
| 80.000 | 80.000.000 VND |

Ứng dụng thực hiện quy đổi để hiển thị giá theo VND cho người dùng. Dữ liệu đầu vào và đầu ra của mô hình vẫn được giữ theo đơn vị gốc để bảo đảm tính nhất quán khi dự đoán.

## 5. Các mô hình Machine Learning

Dự án sử dụng ba thuật toán hồi quy:

1. **Linear Regression** – Hồi quy tuyến tính.
2. **Decision Tree Regression** – Cây quyết định hồi quy.
3. **Random Forest Regression** – Rừng ngẫu nhiên hồi quy.

Random Forest Regression được chọn làm mô hình mặc định do đạt kết quả tốt nhất trên tập kiểm tra trong lần huấn luyện hiện tại.

## 6. Cấu trúc dự án

```text
PhoneValueAI_Package/
│
├── app/
│   ├── main.py
│   ├── database.py
│   └── pages/
│       ├── 1_Dinh_gia.py
│       ├── 2_Phan_tich.py
│       ├── 3_Lich_su.py
│       └── 4_Thong_tin_mo_hinh.py
│
├── data/
│   ├── raw/
│   │   └── used_phone_price_prediction_1M.csv.gz
│   └── processed/
│       └── feature_importance.csv
│
├── models/
│   ├── linear_regression.pkl
│   ├── decision_tree_regression.pkl
│   ├── random_forest_regression.pkl
│   └── model_metadata.json
│
├── notebooks/
│
├── reports/
│   ├── data_summary.json
│   ├── feature_importance.png
│   ├── missing_values.csv
│   ├── numeric_summary.csv
│   ├── model_overfitting_evaluation.csv
│   ├── resale_price_greater_than_original.csv
│   └── test_training_all_results.csv
│
├── src/
│   ├── inspect_data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── feature_importance.py
│   ├── test_all_predictions.py
│   ├── test_training.py
│   └── test_training_all.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

Ngoài ra, ứng dụng có thể tạo file `phonevalue_history.db` để lưu lịch sử định giá trên máy chạy ứng dụng. File database này không được đưa lên GitHub.

## 7. Cài đặt dự án

### Bước 1: Cài đặt Python

Khuyến nghị sử dụng Python 3.13.7.

Kiểm tra phiên bản:

```powershell
python --version
```

### Bước 2: Tạo môi trường ảo

```powershell
python -m venv .venv
```

### Bước 3: Kích hoạt môi trường ảo

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn thực thi script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Bước 4: Cài đặt thư viện

```powershell
pip install -r requirements.txt
```

### Bước 5: Kiểm tra thư viện

```powershell
pip check
```

## 8. Huấn luyện mô hình

Đảm bảo dataset nằm đúng vị trí:

```text
data/raw/used_phone_price_prediction_1M.csv.gz
```

Chạy lệnh:

```powershell
python src/train.py
```

Sau khi huấn luyện thành công, các mô hình và metadata được lưu trong thư mục `models/`.

Các file mô hình:

- `linear_regression.pkl`
- `decision_tree_regression.pkl`
- `random_forest_regression.pkl`
- `model_metadata.json`

Lưu ý: Huấn luyện trên toàn bộ 1.000.000 dòng có thể mất nhiều thời gian và cần đủ bộ nhớ.

## 9. Dự đoán giá điện thoại

Chạy chương trình kiểm tra dự đoán:

```powershell
python src/predict.py
```

Ứng dụng sử dụng các mô hình đã huấn luyện để dự đoán giá bán lại dựa trên thông tin điện thoại đầu vào.

## 10. Đánh giá mô hình

Chạy:

```powershell
python src/evaluate.py
```

Các chỉ số đánh giá:

- **MAE (Mean Absolute Error):** Sai số tuyệt đối trung bình giữa giá trị dự đoán và giá trị thực tế.
- **RMSE (Root Mean Squared Error):** Căn bậc hai của sai số bình phương trung bình, nhạy hơn với các sai số lớn.
- **R² (R-squared):** Hệ số xác định, phản ánh mức độ giải thích biến thiên của dữ liệu.

## 11. Phân tích độ quan trọng đặc trưng

Chạy:

```powershell
python src/feature_importance.py
```

Kết quả phân tích được lưu tại:

```text
data/processed/feature_importance.csv
reports/feature_importance.png
```

Một số đặc trưng có ảnh hưởng đáng kể trong kết quả phân tích hiện tại:

1. Giá mua ban đầu (`original_price`).
2. Tuổi đời thiết bị (`age_months`).
3. Điểm nhu cầu thị trường (`market_demand_score`).
4. Tình trạng pin (`battery_health`).
5. Tình trạng thiết bị (`condition`).

## 12. Khởi chạy ứng dụng

Tại thư mục gốc dự án, chạy:

```powershell
streamlit run app/main.py --server.port 8502
```

Sau khi khởi chạy thành công, mở trình duyệt tại:

```text
http://localhost:8502
```

## 13. Các chức năng chính

### 13.1. Định giá điện thoại

- Nhập thông tin điện thoại.
- Chọn mô hình Machine Learning để dự đoán.
- Hiển thị giá bán lại dự đoán theo VND.
- So sánh giá mua ban đầu và giá dự đoán.
- Lưu kết quả vào lịch sử định giá.

### 13.2. Phân tích dữ liệu

- Thống kê tổng quan dataset.
- Phân tích giá điện thoại.
- Phân tích theo thương hiệu và các thuộc tính.
- Hiển thị biểu đồ trực quan.
- Hỗ trợ xuất dữ liệu phân tích.

### 13.3. Lịch sử định giá

- Xem các lần định giá trước đó.
- Tìm kiếm và lọc lịch sử.
- Xem chi tiết từng lần định giá.
- Xuất dữ liệu lịch sử.
- Xóa một bản ghi hoặc xóa toàn bộ lịch sử.

### 13.4. Thông tin mô hình

- Hiển thị ba thuật toán Machine Learning.
- So sánh các chỉ số MAE, RMSE và R².
- Hiển thị mô hình mặc định.
- Trình bày thông tin huấn luyện và các file mô hình.

## 14. Kết quả huấn luyện mô hình

Các mô hình được huấn luyện và đánh giá trên dataset gồm 1.000.000 dòng:

- Tập huấn luyện: 800.000 dòng.
- Tập kiểm tra: 200.000 dòng.
- Số đặc trưng đầu vào: 27.

Kết quả đánh giá:

| Mô hình | MAE (VND) | RMSE (VND) | R² |
|---|---:|---:|---:|
| Linear Regression | 4.794.880 | 6.825.660 | 0,833181 |
| Decision Tree Regression | 3.199.630 | 4.325.120 | 0,933019 |
| Random Forest Regression | 2.755.620 | 3.728.630 | 0,950220 |

*MAE và RMSE trong bảng đã được quy đổi từ đơn vị nghìn VND sang VND.*

### Nhận xét

- Linear Regression có kết quả thấp hơn hai mô hình còn lại trong lần đánh giá này.
- Decision Tree Regression cải thiện đáng kể khả năng dự đoán.
- Random Forest Regression đạt MAE và RMSE thấp nhất, đồng thời có R² cao nhất.
- Vì vậy, Random Forest Regression được chọn làm mô hình mặc định của ứng dụng.

## 15. Kết luận

PhoneValue AI là dự án ứng dụng Machine Learning vào bài toán dự đoán giá bán lại điện thoại đã qua sử dụng.

Dự án kết hợp các bước xử lý dữ liệu, huấn luyện và so sánh mô hình, đánh giá hiệu suất, phân tích đặc trưng, trực quan hóa dữ liệu và xây dựng ứng dụng web bằng Streamlit.

Thông qua ba thuật toán hồi quy, dự án hướng đến việc cung cấp công cụ tham khảo giá điện thoại cũ trực quan và thuận tiện cho người dùng.

---

**PhoneValue AI – Used Phone Price Prediction with Machine Learning.**