# Nhật ký đồ án

## 11/11/2026

- Hoàn thiện EDA chính thức cho lần chạy baseline đầu tiên từ dữ liệu nguồn và log lịch sử ngày 28/10/2026.
- Bổ sung từ điển dữ liệu, kiểm tra nhãn, phân tích IQR, tỷ lệ lỗi theo Type với khoảng Wilson 95%, phân tích quan hệ đặc trưng và 5 ma trận nhầm lẫn validation.
- Gom mã EDA, bảng, 10 hình, nhận xét và thông tin tái lập vào một notebook đã thực thi. Xóa script tạo báo cáo và các file EDA xuất phụ.
- Đối chiếu precision, recall, F1 bằng TN/FP/FN/TP và kiểm tra số lượng mẫu validation với phép chia seed 42.
- Chạy Run All từ thư mục `notebooks`, xác nhận đủ 10 hình, 10 bảng và toàn bộ cell thực thi thành công. Dữ liệu, phép chia và log baseline khớp với commit lịch sử `8bb8490`.

**Minh chứng** nằm trong [notebook EDA](notebooks/02_eda_lan_chay_dau.ipynb).

## 06/11/2026

**Đã làm**

- Chạy lại 17 mô hình: 5 baseline E00 và 12 mô hình E01–E03.
- Nạp lại các mô hình đã lưu và đối chiếu dự đoán, độ đo; 6/6 kiểm thử đạt.
- Làm gọn mã, tích hợp tạo đặc trưng vào pipeline và dọn các file trùng.
- Cập nhật README, notebook, Word và slide; áp dụng bản mới vào repository.

**Kết quả:** Gradient Boosting E01/E03 đạt F1 0,9149, Recall 0,8431, bỏ sót 8 lỗi và có 0 báo động giả trên validation. Test chưa đánh giá.

**Minh chứng:** [Log 17 mô hình](results/final/experiment_log.csv), [kiểm chứng mô hình](results/final/verification.json), [đối chiếu lịch sử](results/final/historical_comparison.json), [kiểm thử](tests/test_contracts.py), [Word](reports/BaoCaoGiuaKy_AI4I_2020.docx), [slide](reports/BaoCaoGiuaKy_AI4I_2020.pptx).

## 04/11/2026

**Đã làm**

- Chạy ba đợt E01, E02 và E03 với bốn thuật toán, tổng cộng 12 mô hình.
- Thêm chênh lệch nhiệt độ, công suất ước tính và tương tác mòn × mô-men.
- Tìm tham số ở E02/E03 bằng CV phân tầng 3 fold trên train.
- Phân tích FN/FP và xuất hình minh chứng cho từng thuật toán.

**Kết quả:** Gradient Boosting có F1 validation lần lượt 0,9149 (E01), 0,7391 (E02) và 0,9149 (E03).

**Minh chứng lịch sử:** [Log ngày 04/11](experiments/history/three_rounds_2026-11-04.csv). Hình minh chứng đã tái lập trong bản hiện tại: [cấu trúc mô hình và importance](results/final/algorithm_evidence/).

## 02/11/2026

**Đã làm**

- Kiểm tra mã và notebook sau khi merge Git, kiểm tra dấu xung đột và cú pháp.
- Sửa đường dẫn CSV mặc định và chạy lại năm mô hình để đối chiếu.
- Xác nhận kết quả baseline không thay đổi sau merge.

**Kết quả:** Gradient Boosting giữ F1 0,7059, bỏ sót 21 lỗi và có 4 báo động giả trên validation.

**Minh chứng lịch sử:** Commit `24fad28` ghi bản sửa sau merge. Các file của bản cũ đã được thay khi làm gọn; mã hiện hành nằm tại [data.py](ai4i/data.py) và [cli.py](ai4i/cli.py).

## 28/10/2026

**Đã làm**

- Khám phá dữ liệu AI4I 2020: 10.000 mẫu, 339 mẫu lỗi, không thiếu và không trùng toàn dòng.
- Tiền xử lý bằng encoding và scaling trong pipeline, chia train/validation/test 70/15/15.
- Chạy Dummy, Logistic Regression, Decision Tree, Random Forest và Gradient Boosting làm baseline.
- Tạo README, notebook và tổ chức repository.

**Kết quả:** Gradient Boosting có F1 validation cao nhất trong baseline: 0,7059.

**Minh chứng lịch sử:** [Log năm mô hình baseline](experiments/history/comparison_5_models.csv), [dữ liệu nguồn](data/ai4i2020.csv). Bản phân tích và notebook hiện hành: [EDA](results/final/eda/), [notebook](notebooks/01_phan_tich_va_ket_qua.ipynb).
