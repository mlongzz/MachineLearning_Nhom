# Kết quả bản nộp giữa kỳ

Toàn bộ độ đo bên dưới là validation. Test chưa được đánh giá.

| Đợt | Mô hình | Precision | Recall | F1 | ROC-AUC | AP | FN | FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E00 | Dummy | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0340 | 51 | 0 |
| E00 | LogisticRegression | 0.1365 | 0.7255 | 0.2298 | 0.8750 | 0.3829 | 14 | 234 |
| E00 | DecisionTree | 0.3304 | 0.7255 | 0.4540 | 0.7421 | 0.3763 | 14 | 75 |
| E00 | RandomForest | 0.7027 | 0.5098 | 0.5909 | 0.9739 | 0.6389 | 25 | 11 |
| E00 | GradientBoosting | 0.8824 | 0.5882 | 0.7059 | 0.9772 | 0.8094 | 21 | 4 |
| E01 | LogisticRegression | 0.1695 | 0.7843 | 0.2787 | 0.9048 | 0.3561 | 11 | 196 |
| E01 | DecisionTree | 0.4404 | 0.9412 | 0.6000 | 0.9496 | 0.8002 | 3 | 61 |
| E01 | RandomForest | 0.9268 | 0.7451 | 0.8261 | 0.9801 | 0.8721 | 13 | 3 |
| E01 | GradientBoosting | 1.0000 | 0.8431 | 0.9149 | 0.9826 | 0.9229 | 8 | 0 |
| E02 | LogisticRegression | 0.1370 | 0.7255 | 0.2305 | 0.8749 | 0.3848 | 14 | 233 |
| E02 | DecisionTree | 0.4583 | 0.6471 | 0.5366 | 0.8153 | 0.4889 | 18 | 39 |
| E02 | RandomForest | 0.7105 | 0.5294 | 0.6067 | 0.9690 | 0.6422 | 24 | 11 |
| E02 | GradientBoosting | 0.8293 | 0.6667 | 0.7391 | 0.9836 | 0.8068 | 17 | 7 |
| E03 | LogisticRegression | 0.1681 | 0.7843 | 0.2768 | 0.9037 | 0.3465 | 11 | 198 |
| E03 | DecisionTree | 0.4731 | 0.8627 | 0.6111 | 0.9264 | 0.7365 | 7 | 49 |
| E03 | RandomForest | 0.9268 | 0.7451 | 0.8261 | 0.9801 | 0.8721 | 13 | 3 |
| E03 | GradientBoosting | 1.0000 | 0.8431 | 0.9149 | 0.9816 | 0.9234 | 8 | 0 |

Gradient Boosting E01: F1 0.9149, Recall 0.8431, TP 43, FN 8, FP 0.

E01 và E03 của Gradient Boosting có cùng F1 trên lần chia này. Đây là ứng viên giữa kỳ, chưa phải xác nhận trên test.
Dữ liệu mô phỏng, nhãn có bất nhất và validation chỉ có 51 mẫu lỗi. Chạy lại cùng seed kiểm tra tái lập, không bổ sung bằng chứng độc lập.

Permutation importance trong algorithm_evidence đo riêng các cột sau tạo đặc trưng, giữ cùng giao thức với báo cáo. Không diễn giải là quan hệ nhân quả.
