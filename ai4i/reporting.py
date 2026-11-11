"""Export model evidence and comparison figures from fitted pipelines."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import export_text
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss
from .config import BASE
from .data import augment
from .storage import dump
from .figures import make_figures

def export_evidence(dataset, logs, output_dir):
    """Inspect validation predictions; test data is never evaluated."""
    out = Path(output_dir)
    X, y, iv = dataset.features, dataset.target, dataset.validation_indices
    ev = out / 'algorithm_evidence'
    ev.mkdir(exist_ok=True)
    for name in BASE:
        best = logs[logs.model == name].sort_values(['f1', 'run'], ascending=[False, True]).iloc[0]
        run = best['run']
        target = ev / name
        target.mkdir(exist_ok=True)
        pipe = joblib.load(out / run / name / 'model.joblib')
        xx = augment(X) if best['engineered_features'] else X.copy()
        # Inspect the already engineered columns independently, matching the report.
        # End-to-end inference still uses the complete raw-input pipeline.
        inspected = Pipeline(pipe.steps[1:]) if 'features' in pipe.named_steps else pipe
        model = pipe.named_steps['model']
        xt = pipe.named_steps['preprocess'].transform(xx.loc[iv])
        feat = pipe.named_steps['preprocess'].get_feature_names_out()
        dump(target / 'candidate.json', dict(run=run, selection='highest validation F1 among E01-E03, not test-confirmed', parameters=model.get_params(), importance_scope='conditional permutation after feature construction; not raw-input causal effects'))
        perm = permutation_importance(inspected, xx.loc[iv], y.loc[iv], scoring='f1', n_repeats=5, random_state=42, n_jobs=1)
        imp = pd.DataFrame({'feature': xx.columns, 'mean_f1_decrease': perm.importances_mean, 'std': perm.importances_std}).sort_values('mean_f1_decrease')
        imp.to_csv(target / 'permutation_importance.csv', index=False)
        ax = imp.plot.barh(x='feature', y='mean_f1_decrease', xerr='std', figsize=(10, 6), legend=False)
        ax.set_title(f'{name} {run}: permutation importance on validation')
        plt.tight_layout()
        plt.savefig(target / 'permutation_importance.png', dpi=180)
        plt.close()
        if name == 'LogisticRegression':
            cf = pd.DataFrame({'feature': feat, 'coefficient': model.coef_[0]})
            cf.to_csv(target / 'coefficients.csv', index=False)
            dump(target / 'intercept.json', model.intercept_.tolist())
            cf.sort_values('coefficient').plot.barh(x='feature', y='coefficient', figsize=(10, 6), legend=False)
            plt.title(f'Logistic Regression {run}: standardized coefficients')
            plt.tight_layout()
            plt.savefig(target / 'model_structure.png', dpi=180)
            plt.close()
        elif name == 'DecisionTree':
            (target / 'tree_rules.txt').write_text(export_text(model, feature_names=list(feat)), encoding='utf-8')
            dump(target / 'tree_summary.json', dict(n_nodes=model.tree_.node_count, depth=model.get_depth(), leaves=model.get_n_leaves()))
        elif name == 'RandomForest':
            trees = pd.DataFrame([dict(tree=i + 1, depth=t.get_depth(), leaves=t.get_n_leaves(), nodes=t.tree_.node_count) for i, t in enumerate(model.estimators_)])
            trees.to_csv(target / 'forest_structure.csv', index=False)
        else:
            train_losses = model.train_score_
            val_losses = [log_loss(y.loc[iv], pp) for pp in pipe.named_steps['model'].staged_predict_proba(xt)]
            pd.DataFrame({'iteration': np.arange(1, len(train_losses) + 1), 'train_log_loss': train_losses, 'validation_log_loss': val_losses}).to_csv(target / 'boosting_loss.csv', index=False)
            plt.figure(figsize=(9, 5))
            plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label='Train')
            plt.plot(np.arange(1, len(val_losses) + 1), val_losses, label='Validation')
            plt.xlabel('Boosting iteration')
            plt.ylabel('Log loss')
            plt.legend()
            plt.title(f'Gradient Boosting {run}: observed loss trajectory')
            plt.tight_layout()
            plt.savefig(target / 'model_structure.png', dpi=180)
            plt.close()
    for run in ['E01', 'E02', 'E03']:
        part = logs[logs.run == run]
        part.set_index('model')[['precision', 'recall', 'f1', 'roc_auc', 'average_precision']].plot.bar(figsize=(12, 6), ylim=(0, 1.05))
        plt.xticks(rotation=15, ha='right')
        plt.title(f'{run}: validation metrics')
        plt.tight_layout()
        plt.savefig(out / run / 'comparison.png', dpi=180)
        plt.close()
    pivot = logs.pivot(index='model', columns='run', values='f1')
    pivot.plot.bar(figsize=(11, 6), ylim=(0, 1.05))
    plt.title('Three experiment rounds: validation F1')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(out / 'rounds_f1.png', dpi=180)
    plt.close()
    make_figures(out)
