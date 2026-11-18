

# PhoneValue AI – Hệ thống định giá điện thoại cũ bằng Machine Learning



## 1. Giới thiệu



PhoneValue AI là ứng dụng sử dụng Machine Learning để dự đoán giá bán lại của điện thoại đã qua sử dụng.



Hệ thống cho phép người dùng nhập thông tin điện thoại, nhận kết quả định giá, phân tích dữ liệu, xem lịch sử định giá và tìm hiểu thông tin về mô hình dự đoán.



## 2. Mục tiêu dự án



\- Xây dựng mô hình Machine Learning dự đoán giá bán lại điện thoại cũ.

\- Phân tích các yếu tố ảnh hưởng đến giá trị điện thoại.

\- Xây dựng ứng dụng web thân thiện, dễ sử dụng.

\- Lưu trữ và quản lý lịch sử định giá.

\- Đánh giá hiệu quả của mô hình thông qua các chỉ số MAE, RMSE và R².



## 3. Công nghệ sử dụng



| Công nghệ | Mục đích |

|---|---|

| Python 3.13.7 | Ngôn ngữ lập trình |

| Pandas, NumPy | Xử lý dữ liệu |

| Scikit-learn | Xây dựng mô hình Machine Learning |

| HistGradientBoostingRegressor | Thuật toán dự đoán |

| Matplotlib, Seaborn, Plotly | Trực quan hóa dữ liệu |

| Streamlit | Xây dựng giao diện web |

| Joblib | Lưu và tải mô hình |

| SQLite | Lưu lịch sử định giá |



## 4. Dữ liệu



Dự án sử dụng bộ dữ liệu điện thoại cũ gồm:



\- 1.000.000 dòng dữ liệu.

\- 27 đặc trưng đầu vào.

\- 1 biến mục tiêu: `resale\_price`.



Một số thông tin đầu vào:



\- Hãng và model điện thoại.

\- Năm phát hành.

\- RAM và dung lượng lưu trữ.

\- Dung lượng pin và tình trạng pin.

\- Tình trạng ngoại hình.

\- Lịch sử sửa chữa.

\- Tình trạng bảo hành.

\- Nhu cầu thị trường.



\*\*Lưu ý:\*\* Đơn vị giá được giữ theo dữ liệu gốc, không tự quy đổi thành VND khi chưa xác định được đơn vị của bộ dữ liệu.



## 5. Cấu trúc dự án



```text

UsedPhonePrice/

│

├── app/

│   ├── main.py

│   ├── database.py

│   └── pages/

│       ├── 1\_Dinh\_gia.py

│       ├── 2\_Phan\_tich.py

│       ├── 3\_Lich\_su.py

│       └── 4\_Thong\_tin\_mo\_hinh.py

│

├── data/

│   ├── raw/

│   │   └── used\_phone\_price\_prediction\_1M.csv

│   └── processed/

│       └── feature\_importance.csv

│

├── models/

│   ├── best\_model.pkl

│   └── model\_metadata.json

│

├── reports/

│   ├── data\_summary.json

│   ├── feature\_importance.png

│   ├── missing\_values.csv

│   └── numeric\_summary.csv

│

├── src/

│   ├── inspect\_data.py

│   ├── preprocessing.py

│   ├── train.py

│   ├── predict.py

│   ├── evaluate.py

│   └── feature\_importance.py

│

├── requirements.txt

└── README.md

```



## 6. Cài đặt dự án



### Bước 1: Cài đặt Python



Khuyến nghị sử dụng Python 3.13.7.



### Bước 2: Tạo môi trường ảo



```powershell

python -m venv .venv

```



### Bước 3: Kích hoạt môi trường ảo



Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Nếu bị chặn quyền chạy script:



```powershell

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\\.venv\\Scripts\\Activate.ps1

```



### Bước 4: Cài đặt thư viện



```powershell

pip install -r requirements.txt

```



### Bước 5: Kiểm tra thư viện



```powershell

pip check

```



## 7. Huấn luyện mô hình



Đảm bảo bộ dữ liệu đã nằm tại:



```text

data/raw/used\_phone\_price\_prediction\_1M.csv

```



Chạy lệnh:



```powershell

python src/train.py

```



Sau khi huấn luyện thành công, mô hình và thông tin đánh giá được lưu trong thư mục `models/`.



## 8. Dự đoán giá điện thoại



Chạy chương trình kiểm tra dự đoán:



```powershell

python src/predict.py

```



## 9. Đánh giá mô hình



Chạy:



```powershell

python src/evaluate.py

```



Các chỉ số đánh giá:



\- MAE: Sai số tuyệt đối trung bình.

\- RMSE: Căn bậc hai của sai số bình phương trung bình.

\- R²: Hệ số xác định, phản ánh mức độ giải thích biến thiên của dữ liệu.



## 10. Phân tích độ quan trọng đặc trưng



Chạy:



```powershell

python src/feature\_importance.py

```



Kết quả được lưu tại:



```text

data/processed/feature\_importance.csv

reports/feature\_importance.png

```



## 11. Khởi chạy ứng dụng



Tại thư mục gốc của dự án, chạy:



```powershell

streamlit run app/main.py

```



Sau đó mở địa chỉ Local URL mà Streamlit hiển thị trên terminal, thông thường là:



```text

http://localhost:8501

```



## 12. Các chức năng chính



### 12.1. Định giá điện thoại



\- Nhập thông tin điện thoại.

\- Dự đoán giá bán lại bằng mô hình Machine Learning.

\- Hiển thị kết quả định giá.

\- Lưu kết quả vào lịch sử.



### 12.2. Phân tích dữ liệu



\- Thống kê dữ liệu điện thoại.

\- Phân tích các thuộc tính.

\- Trực quan hóa dữ liệu bằng biểu đồ.



### 12.3. Lịch sử định giá



\- Xem các lần định giá trước đó.

\- Tìm kiếm và lọc lịch sử.

\- Xuất dữ liệu lịch sử.

\- Xóa bản ghi hoặc xóa toàn bộ lịch sử.



### 12.4. Thông tin mô hình



\- Hiển thị thuật toán Machine Learning.

\- Hiển thị các chỉ số đánh giá.

\- Hiển thị độ quan trọng của đặc trưng.

\- Cung cấp thông tin về quá trình huấn luyện.



## 13. Kết quả đánh giá mô hình



Mô hình HistGradientBoostingRegressor được đánh giá trên tập kiểm tra gồm 200.000 dòng.



Kết quả huấn luyện đã ghi nhận:



| Chỉ số | Giá trị |

|---|---:|

| MAE | 1.801,63 |

| RMSE | 2.297,59 |

| R² | 0,9811 |

| Số dòng huấn luyện | 800.000 |

| Số dòng kiểm tra | 200.000 |

| Số đặc trưng | 27 |



Kết quả kiểm tra bổ sung trên mẫu 20.000 dòng:



| Chỉ số | Tập huấn luyện | Tập kiểm tra |

|---|---:|---:|

| MAE | 1.783,51 | 1.807,77 |

| RMSE | 2.266,58 | 2.294,56 |

| R² | 0,9818 | 0,9813 |



Các chỉ số trên cho thấy hiệu suất giữa hai tập khá gần nhau trong lần đánh giá mẫu này.



## 14. Kết luận



PhoneValue AI là ứng dụng minh họa việc áp dụng Machine Learning vào bài toán dự đoán giá bán lại điện thoại đã qua sử dụng.



Dự án kết hợp xử lý dữ liệu, huấn luyện mô hình, đánh giá hiệu suất, trực quan hóa và xây dựng ứng dụng web bằng Streamlit.




