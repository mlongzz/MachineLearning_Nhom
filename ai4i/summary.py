"""Readable final tables, charts and an explicitly validation-only conclusion."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


def export_summary(logs, output):
    out = Path(output)
    comparison = logs[["run", "model", "precision", "recall", "f1", "roc_auc", "average_precision", "tp", "fn", "fp", "tn"]]
    comparison.to_csv(out / "model_comparison.csv", index=False)
    pivot = logs.pivot(index="model", columns="run", values="f1").reindex(columns=["E00", "E01", "E02", "E03"])
    pivot.to_csv(out / "f1_by_round.csv")
    ax = pivot.drop(index="Dummy").plot.bar(figsize=(11, 5), ylim=(0, 1.05))
    ax.set(title="Validation F1: E00 to E03", ylabel="F1 of failure class")
    plt.xticks(rotation=15, ha="right"); plt.tight_layout()
    plt.savefig(out / "all_rounds_f1.png", dpi=180); plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in logs[logs.run == "E01"].model:
        pred = pd.read_csv(out / "E01" / name / "validation_predictions.csv")
        fpr, tpr, _ = roc_curve(pred.actual, pred.failure_probability)
        precision, recall, _ = precision_recall_curve(pred.actual, pred.failure_probability)
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(recall, precision, label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].axhline(0.034, ls="--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="Recall", title="E01 ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="E01 Precision-Recall")
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(out / "roc_pr_e01.png", dpi=180); plt.close(fig)
    best = logs[logs.model == "GradientBoosting"].sort_values(["f1", "run"], ascending=[False, True]).iloc[0]
    lines = ["# Kết quả bản nộp giữa kỳ", "", "Toàn bộ độ đo bên dưới là validation. Test chưa được đánh giá.", "",
             "| Đợt | Mô hình | Precision | Recall | F1 | ROC-AUC | AP | FN | FP |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in logs.itertuples():
        lines.append(f"| {r.run} | {r.model} | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} | {r.roc_auc:.4f} | {r.average_precision:.4f} | {r.fn} | {r.fp} |")
    lines += ["", f"Gradient Boosting {best.run}: F1 {best.f1:.4f}, Recall {best.recall:.4f}, TP {int(best.tp)}, FN {int(best.fn)}, FP {int(best.fp)}.",
              "", "E01 và E03 của Gradient Boosting có cùng F1 trên lần chia này. Đây là ứng viên giữa kỳ, chưa phải xác nhận trên test.",
              "Dữ liệu mô phỏng, nhãn có bất nhất và validation chỉ có 51 mẫu lỗi. Chạy lại cùng seed kiểm tra tái lập, không bổ sung bằng chứng độc lập.",
              "", "Permutation importance trong algorithm_evidence đo riêng các cột sau tạo đặc trưng, giữ cùng giao thức với báo cáo. Không diễn giải là quan hệ nhân quả."]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
