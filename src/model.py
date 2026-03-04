import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def build_pipeline():
    # Using 'balanced' is excellent for medical data (handling class imbalance)
    logreg = LogisticRegression(
        solver='lbfgs',
        class_weight='balanced',
        max_iter=1000
    )

    pipe = Pipeline([
        ('rfe', RFE(estimator=LogisticRegression())),
        ('smote', SMOTE(random_state=42)),
        ('logreg', logreg)
    ])
    return pipe

def tune_model(pipe, X_train, y_train):
    param_grid = {
        'rfe__n_features_to_select': [10],
        'logreg__C': [0.1, 1, 10]
    }

    # StratifiedKFold is perfect for medical research to maintain class ratios
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring='f1_macro', # Highlighting macro-F1 is great for imbalanced data
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_
