# AI4I — Kết quả so sánh mô hình

Chia phân tầng 70/15/15, seed 42; chỉ đánh giá validation. Tập test chưa sử dụng.
Các cấu hình ban đầu chưa được tối ưu. Mỗi mô hình dùng ngưỡng/quy tắc predict mặc định.

| Mô hình | Precision | Recall | F1 | ROC-AUC | AP | Bỏ sót | Báo động giả |
|---|---:|---:|---:|---:|---:|---:|---:|
| GradientBoosting | 0.8824 | 0.5882 | 0.7059 | 0.9772 | 0.8094 | 21 | 4 |
| RandomForest | 0.7027 | 0.5098 | 0.5909 | 0.9739 | 0.6389 | 25 | 11 |
| DecisionTree | 0.3304 | 0.7255 | 0.4540 | 0.7421 | 0.3763 | 14 | 75 |
| LogisticRegression | 0.1365 | 0.7255 | 0.2298 | 0.8750 | 0.3829 | 14 | 234 |
| Dummy | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0340 | 51 | 0 |

AP là Average Precision, không phải tích phân hình thang của đường PR.
Kết quả phụ thuộc tập validation nhỏ và dữ liệu mô phỏng; không chứng minh dự đoán sớm thời điểm hỏng.
Bước tiếp theo: cross-validation trên train, thử đặc trưng mới và chọn ngưỡng trên validation; sau khi chốt mới đánh giá test.