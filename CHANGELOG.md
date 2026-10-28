# Nhật ký đồ án

**AI4I 2020 · Dự đoán lỗi thiết bị**  
Ghi theo ngày Việt Nam (UTC+7). Cập nhật mới đặt ở trên, giữ lại lịch sử cũ.

---

## 28/10/2026 · Lần 02
### Hoàn thiện tài liệu & tổ chức dự án

**Người thực hiện:** Chưa điền tên thành viên.

| Đã thực hiện | Kết quả |
|---|---|
| Tổng hợp README | Có giới thiệu, cách chạy, bảng kết quả và biểu đồ |
| Cập nhật notebook | Dùng đường dẫn tương đối |
| Dọn thư mục | Xóa 3 bản sao lưu và README trùng; thêm `.gitignore` |
| Chuẩn hóa nhật ký | Có ngày, số lần cập nhật và mẫu dùng lại |

**Minh chứng:** [README](README.md) · [Notebook](01_eda_baseline.ipynb) · [.gitignore](.gitignore)

**Còn lại:** Điền đóng góp thành viên, kiểm tra và push lên GitHub. Chưa xác nhận đã push. Không có thực nghiệm mới trong lần cập nhật này.

---

## 28/10/2026 · Lần 01
### EDA & so sánh 5 mô hình

**Người thực hiện:** Chưa điền tên thành viên.  
**Thiết lập:** Train/validation/test = **70/15/15** · Seed **42** · **Chưa đánh giá test**.

**Đã làm**

- Kiểm tra **10.000 mẫu**, không thiếu và không trùng; **339 mẫu lỗi (3,39%)**.
- Vẽ EDA; encoding/scaling chỉ fit trên train trong pipeline.
- Chạy Dummy, Logistic Regression, Decision Tree, Random Forest và Gradient Boosting.
- Lưu tham số, metrics, dự đoán, confusion matrix và phiên bản môi trường.

**Kết quả nổi bật trên validation**

| Mô hình | Precision | Recall | F1 | Bỏ sót / 51 lỗi | Báo động giả |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0,8824 | 0,5882 | **0,7059** | 21 | 4 |
| Decision Tree | 0,3304 | **0,7255** | 0,4540 | 14 | 75 |

Gradient Boosting có F1 cao nhất lần chạy; Decision Tree phát hiện nhiều lỗi hơn nhưng cảnh báo sai nhiều hơn. Các chỉ số trên tính cho lớp lỗi.

**Minh chứng:** [Nhật ký thực nghiệm](results/comparison_5_models/experiment_log.csv) · [Bảng đủ 5 mô hình](results/comparison_5_models/model_comparison.csv) · [Biểu đồ](results/comparison_5_models/model_comparison.png) · [Nhận xét](results/comparison_5_models/NHAN_XET.md)

**Hạn chế:** Chưa tối ưu tham số/ngưỡng; dữ liệu mô phỏng, validation chỉ có 51 mẫu lỗi. Phạm vi hiện tại là phân loại trạng thái lỗi, chưa dự đoán thời điểm hỏng hay lịch bảo trì. Mục này tổng hợp công việc trong ngày; không có giờ chạy chính xác.

**Tiếp theo:** Thử đặc trưng mới, cross-validation trên train, chọn ngưỡng trên validation; hoàn thiện báo cáo, slide và phân công thực tế.

---

<details>
<summary><b>Mẫu cập nhật mới — bấm để mở</b></summary>

Sao chép mẫu và chèn trên mục mới nhất. Chỉ thêm một phần cho mỗi đợt cập nhật; không sửa lịch sử để thay kết quả cũ. Nếu cần đính chính, ghi trong mục mới.

```markdown
## DD/MM/YYYY · Lần 01
### Tên công việc

**Người thực hiện:** Họ tên và phần việc.

**Đã làm**

- Công việc 1.
- Công việc 2.

**Kết quả:** Số liệu thực tế hoặc tài liệu hoàn thành.

**Minh chứng:** [Kết quả](results/YYYY-MM-DD_run_01/experiment_log.csv)

**Hạn chế:** Vấn đề còn tồn tại; nếu không chạy mô hình thì ghi rõ.

**Tiếp theo:** Công việc và người phụ trách.

---
```

Mỗi lần chạy dùng thư mục riêng để giữ kết quả cũ:

```powershell
python ai4i_eda_baseline.py --output results/YYYY-MM-DD_run_01
```

Thay `YYYY-MM-DD` bằng ngày chạy thực tế. Chương trình tự xuất `experiment_log.csv`; phần nhật ký này được bổ sung thủ công. Không dùng lại cùng thư mục output nếu muốn giữ lịch sử.

</details>
