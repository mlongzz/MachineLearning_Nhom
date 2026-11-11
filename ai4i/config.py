"""Fixed experiment settings. Changes belong in a new output directory."""

NUM = ['Air temperature [K]',
 'Process temperature [K]',
 'Rotational speed [rpm]',
 'Torque [Nm]',
 'Tool wear [min]']

FE = ['Temperature difference [K]', 'Estimated power [W]', 'Wear torque [minNm]']

SEED = 42

BASE = {'LogisticRegression': {'max_iter': 1000,
                        'class_weight': 'balanced',
                        'random_state': 42},
 'DecisionTree': {'max_depth': 6,
                  'min_samples_leaf': 5,
                  'class_weight': 'balanced',
                  'random_state': 42},
 'RandomForest': {'n_estimators': 300,
                  'min_samples_leaf': 2,
                  'class_weight': 'balanced',
                  'random_state': 42,
                  'n_jobs': 1},
 'GradientBoosting': {'n_estimators': 100,
                      'learning_rate': 0.1,
                      'max_depth': 3,
                      'random_state': 42}}

GRIDS = {'LogisticRegression': {'model__C': [0.1, 1, 10]},
 'DecisionTree': {'model__max_depth': [4, 6, 10], 'model__min_samples_leaf': [2, 5]},
 'RandomForest': {'model__n_estimators': [150, 300],
                  'model__max_depth': [None, 10],
                  'model__min_samples_leaf': [1, 2]},
 'GradientBoosting': {'model__n_estimators': [100, 200],
                      'model__learning_rate': [0.05, 0.1],
                      'model__max_depth': [2, 3]}}

ROUNDS = (("E01", True, False), ("E02", False, True), ("E03", True, True))
