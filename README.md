# AI4I 2020 — Bộ khởi đầu

## Cách chạy
1. Cài Python 3 và mở terminal trong thư mục đã giải nén.
2. Chạy: `python -m pip install -r requirements.txt`
3. Chạy: `python ai4i_eda_baseline.py --data "D:\đồ án máy học\ai4i2020.csv" --output results`

Hoặc mở `01_eda_baseline.ipynb` bằng Jupyter/VS Code, chọn Python kernel và chạy các ô theo thứ tự. Đặt notebook và file Python cùng thư mục, mở kernel tại thư mục này.

## Kết quả sau khi chạy
Thư mục results chứa thống kê dữ liệu, phân bố nhãn, biểu đồ EDA, confusion matrix, báo cáo phân loại, chỉ số chia dữ liệu và experiment_log.csv.
Nhật ký tự ghi kết quả Dummy và Logistic Regression trên validation; mỗi lần chạy ghi đè cùng thư mục kết quả, vì vậy dùng --output results/run_02 nếu muốn giữ lần chạy cũ.

## Thiết kế
- 6 đầu vào: Type và 5 biến số; nhãn Machine failure.
- Bỏ UDI, Product ID và TWF/HDF/PWF/OSF/RNF khỏi đầu vào.
- Chia phân tầng 70/15/15, seed 42. Chỉ đánh giá validation, chưa dùng test.
- Encoding/scaling fit trên train trong Pipeline.
- Kiểm tra dữ liệu thiếu và dừng nếu có; không tự xóa ngoại lệ hay dòng trùng.
- Average precision là một độ đo tóm tắt đường Precision–Recall, được ghi đúng tên riêng trong nhật ký.
- Dữ liệu mô phỏng, nhãn phản ánh trạng thái lỗi tại điểm đo; không suy diễn thời gian còn lại hay lịch bảo trì tối ưu.

Nguồn: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Bộ này không kèm dataset và chưa chứa kết quả chạy thực nghiệm. Cần chạy trên CSV của bạn trước khi điền số liệu vào báo cáo.
