    """Build an executed EDA notebook and HTML preview for the initial baseline.

Run from the repository root: py notebooks/build_first_run_eda.py
"""
from pathlib import Path
import base64
import io
import json
import sys
import html

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from ai4i.data import load_dataset
from ai4i.config import NUM

OUT = ROOT / 'results' / 'eda_first_run'
OUT.mkdir(parents=True, exist_ok=True)
cells, sections = [], []
scope = {'ROOT': ROOT, 'OUT': OUT, 'pd': pd, 'plt': plt,
         'load_dataset': load_dataset, 'NUM': NUM}

def note(title, text):
    cells.append({'cell_type': 'markdown', 'metadata': {},
                  'source': f'## {title}\n\n{text}'})
    sections.append(f'<h2>{html.escape(title)}</h2><p>{html.escape(text)}</p>')

def run(code, filename=None):
    exec(code, scope)
    outputs = []
    if filename:
        fig = plt.gcf()
        fig.tight_layout()
        path = OUT / filename
        fig.savefig(path, dpi=150, bbox_inches='tight')
        encoded = base64.b64encode(path.read_bytes()).decode()
        outputs.append({'output_type': 'display_data', 'metadata': {},
                        'data': {'image/png': encoded}})
        sections.append(f'<img src="data:image/png;base64,{encoded}" alt="{filename}">')
        plt.close(fig)
    else:
        table = scope['table']
        markup = table.to_html(index=False, float_format=lambda x: f'{x:.4f}')
        outputs.append({'output_type': 'display_data', 'metadata': {},
                        'data': {'text/html': markup, 'text/plain': table.to_string(index=False)}})
        sections.append(markup)
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': code,
                  'execution_count': sum(c['cell_type'] == 'code' for c in cells) + 1,
                  'outputs': outputs})

note('EDA lần chạy đầu tiên — AI4I 2020',
     'Phạm vi: baseline 5 thuật toán ngày 28/10/2026, dùng log lịch sử comparison_5_models.csv. '
     'Đây là EDA được dựng lại từ dữ liệu nguồn hiện tại và phép chia seed 42; không phải bản EDA đã lưu tại thời điểm chạy đầu. '
     'Chỉ khám phá quan hệ trên train; validation chỉ dùng trình bày kết quả baseline đã có. Không đánh giá test, không huấn luyện lại, không dùng kết quả E01–E03.')
run("""from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from ai4i.data import load_dataset
from ai4i.config import NUM
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'ai4i').is_dir())
dataset = load_dataset(ROOT / 'data/ai4i2020.csv')
df = dataset.frame
train = df.loc[dataset.train_indices].copy()
baseline = pd.read_csv(ROOT / 'experiments/history/comparison_5_models.csv')
table = pd.DataFrame([{'Phạm vi': 'Toàn bộ nguồn', 'Số dòng': len(df), 'Số cột': len(df.columns), 'Ô thiếu': int(df.isna().sum().sum()), 'Dòng trùng': int(df.duplicated().sum())}, {'Phạm vi': 'Train dùng cho EDA', 'Số dòng': len(train), 'Số cột': len(train.columns), 'Ô thiếu': int(train.isna().sum().sum()), 'Dòng trùng': int(train.duplicated().sum())}])
""")
note('1. Mất cân bằng nhãn trên train', 'Đọc cả số lượng và tỷ lệ lỗi. Một mô hình luôn dự đoán bình thường có thể đạt accuracy cao nhưng không phát hiện được lỗi nào.')
run("""counts = train['Machine failure'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(['Normal', 'Failure'], counts, color=['#537e99', '#d56a50'])
for i, n in enumerate(counts): ax.text(i, n, f'{n:,} ({n/len(train):.2%})', ha='center', va='bottom')
ax.set(title='Training class distribution', ylabel='Samples', ylim=(0, counts.max()*1.18))
""", '01_class_balance.png')
note('2. Thống kê các biến đầu vào', 'Sử dụng Type và 5 thông số vận hành. UDI/Product ID là định danh; các cờ TWF/HDF/PWF/OSF/RNF không được đưa vào đầu vào dự đoán nhãn tổng.')
run("""table = train[NUM].describe().T.reset_index().rename(columns={'index': 'Feature'})
""")
note('3. Phân bố thông số vận hành', 'Histogram giúp quan sát độ lệch, miền giá trị và các vùng ít mẫu. Không tự xóa điểm cực trị chỉ vì chúng nằm xa phần lớn dữ liệu.')
run("""fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for col, ax in zip(NUM, axes.flat):
    ax.hist(train[col], bins=35, color='#537e99', edgecolor='white')
    ax.set(title=col, ylabel='Samples')
axes.flat[-1].axis('off')
""", '02_distributions.png')
note('4. Máy lỗi khác máy bình thường thế nào?', 'Boxplot so sánh trung vị, độ phân tán và điểm cực trị giữa hai nhóm. Phân bố chồng lấp cho thấy khó phân loại bằng một ngưỡng đơn lẻ.')
run("""fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for col, ax in zip(NUM, axes.flat):
    ax.boxplot([train.loc[train['Machine failure']==c, col] for c in (0,1)], tick_labels=['Normal', 'Failure'], showfliers=False)
    ax.set(title=col)
axes.flat[-1].axis('off')
""", '03_by_failure.png')
note('5. Tỷ lệ lỗi theo Type', 'So sánh tỷ lệ cùng số mẫu từng nhóm; số lỗi tuyệt đối cao có thể do nhóm đó có nhiều quan sát hơn. Đây là liên hệ mô tả, không chứng minh quan hệ nhân quả.')
run("""table = train.groupby('Type')['Machine failure'].agg(['count','sum','mean']).reset_index()
table.columns = ['Type','Samples','Failures','Failure rate']
""")
run("""fig, ax = plt.subplots(figsize=(7,4))
ax.bar(table['Type'], table['Failure rate']*100, color='#537e99')
for i, row in table.iterrows(): ax.text(i, row['Failure rate']*100, f\"{row['Failure rate']:.2%}\\nn={row['Samples']}\", ha='center', va='bottom')
ax.set(ylabel='Failure rate (%)', title='Failure rate by Type — train', ylim=(0, table['Failure rate'].max()*140))
""", '04_type_failure.png')
note('6. Quan hệ giữa thông số', 'Tô màu lỗi trên toàn bộ train, vẽ nhóm lỗi sau để dễ nhìn. Quan sát vùng lỗi ở các tổ hợp tốc độ, mô-men và độ mòn; chưa tạo đặc trưng mới trong EDA baseline.')
run("""fig, axes = plt.subplots(1,2,figsize=(13,5))
for ax, x, y in [(axes[0], NUM[2], NUM[3]), (axes[1], NUM[4], NUM[3])]:
    for label, color, name in [(0,'#9cb4c3','Normal'),(1,'#c84432','Failure')]:
        part = train.loc[train['Machine failure']==label]
        ax.scatter(part[x],part[y],s=10 if label==0 else 22,alpha=.25 if label==0 else .85,c=color,label=name)
    ax.set(xlabel=x,ylabel=y); ax.legend()
""", '05_interactions.png')
note('7. Tương quan tuyến tính', 'Chỉ tính trên 5 biến số đầu vào của train. Tương quan thấp không loại trừ quan hệ phi tuyến hoặc tương tác giữa các biến.')
run("""corr = train[NUM].corr()
fig, ax = plt.subplots(figsize=(8,7))
im = ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
ax.set_xticks(range(5),NUM,rotation=65,ha='right'); ax.set_yticks(range(5),NUM)
for i in range(5):
    for j in range(5): ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center')
fig.colorbar(im,ax=ax)
""", '06_correlation.png')
note('8. Kết quả của lần chạy baseline đầu tiên', 'Nguồn: log lịch sử, đánh giá trên validation. Đây là phần liên hệ mô hình, tách khỏi EDA train. Không trộn kết quả các đợt cải tiến vào lần chạy đầu.')
run("""table = baseline[['model','precision','recall','f1','tn','fp','fn','tp']].copy()
""")
run("""fig, ax = plt.subplots(figsize=(10,4))
baseline.set_index('model')[['precision','recall','f1']].plot.bar(ax=ax,color=['#537e99','#e0a044','#4b9a7d'],rot=15)
ax.set(title='Initial baseline — validation only',ylabel='Score',ylim=(0,1)); ax.legend(loc='upper left')
""", '07_baseline.png')
best = scope['baseline'].sort_values('f1').iloc[-1]
train = scope['train']
note('Nhận xét demo từ số liệu',
     f"Train có {len(train):,} mẫu, {int(train['Machine failure'].sum())} mẫu lỗi ({train['Machine failure'].mean():.2%}). "
     f"Ở lần chạy đầu, {best['model']} có F1 validation cao nhất {best['f1']:.4f}, recall {best['recall']:.4f}; "
     f"bỏ sót {int(best['fn'])} lỗi và có {int(best['fp'])} báo động giả. "
     'Dummy có F1 = 0, nên accuracy đơn lẻ không phù hợp để kết luận chất lượng. '
     'Bước tiếp theo: đối chiếu các vùng lỗi trong scatter/boxplot để đặt giả thuyết đặc trưng và kiểm chứng bằng các thí nghiệm riêng trên train/CV. '
     'Không suy diễn quan hệ nhân quả từ các biểu đồ này.')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
(ROOT / 'notebooks/02_eda_lan_chay_dau.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
preview = '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>EDA lần chạy đầu — AI4I</title><style>body{font:16px/1.65 system-ui;max-width:1080px;margin:40px auto;padding:0 24px;color:#183044;background:#f7f9fc}h2{margin-top:36px;color:#165b72}img{width:100%;background:white;border-radius:10px;margin:18px 0}table{border-collapse:collapse;background:white;font-size:14px}td,th{padding:8px 12px;border:1px solid #dce3ea}p{max-width:950px}</style>'''
(OUT / 'demo.html').write_text(preview + '\n'.join(sections) + '</html>', encoding='utf-8')
print('Created notebooks/02_eda_lan_chay_dau.ipynb and results/eda_first_run/demo.html (7 figures).')
