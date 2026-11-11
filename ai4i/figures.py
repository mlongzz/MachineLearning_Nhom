"""Readable diagrams using actual fitted model structure."""
from pathlib import Path
import pandas as pd,numpy as np,joblib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def make_figures(results):
    r = Path(results)
    plt.rcParams.update({'font.size': 13})
    for name in ['DecisionTree', 'RandomForest']:
        e = r / 'algorithm_evidence' / name
        c = json.loads((e / 'candidate.json').read_text())
        p = joblib.load(r / c['run'] / name / 'model.joblib')
        m = p.named_steps['model']
        features = p.named_steps['preprocess'].get_feature_names_out()
        short = []
        for f in features:
            t = f.replace('num__', '').replace('cat__', '').replace('Temperature difference [K]', 'Delta T [K]').replace('Rotational speed [rpm]', 'Speed [rpm]').replace('Process temperature [K]', 'Process T [K]').replace('Air temperature [K]', 'Air T [K]').replace('Estimated power [W]', 'Power [W]').replace('Wear torque [minNm]', 'Wear x torque')
            short.append(t)
        if name == 'DecisionTree':
            fig = plt.figure(figsize=(9, 5))
            plot_tree(m, feature_names=short, class_names=['no failure', 'failure'], max_depth=1, filled=True, fontsize=12)
            plt.title(f"Decision Tree {c['run']}: root and first child level only")
            plt.tight_layout()
            fig.savefig(e / 'model_structure.png', dpi=220)
            plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.axis('off')
            ax.text(0.5, 0.94, f"Random Forest {c['run']}: {len(m.estimators_)} fitted trees", ha='center', fontsize=17, fontweight='bold')
            for i, x in enumerate([0.17, 0.5, 0.83]):
                tree = m.estimators_[i]
                idx = tree.tree_.feature[0]
                label = f'Tree {i + 1}\nRoot: {short[idx]}\nThreshold: {tree.tree_.threshold[0]:.3f}\nDepth: {tree.get_depth()} | Leaves: {tree.get_n_leaves()}'
                ax.text(x, 0.57, label, ha='center', va='center', fontsize=13, bbox=dict(boxstyle='round,pad=.7', facecolor='#eef3f7', edgecolor='#315f83'))
                ax.annotate('', xy=(0.5, 0.17), xytext=(x, 0.32), arrowprops=dict(arrowstyle='->', color='#315f83'))
            ax.text(0.5, 0.1, 'Mean of tree probabilities -> class decision', ha='center', fontsize=15)
            ax.text(0.5, -0.04, 'Root thresholds refer to the transformed feature scale; only 3 actual trees illustrated.', ha='center', fontsize=10)
            fig.tight_layout()
            fig.savefig(e / 'model_structure.png', dpi=220, bbox_inches='tight')
            plt.close(fig)
    print('Readable algorithm evidence figures updated.')
