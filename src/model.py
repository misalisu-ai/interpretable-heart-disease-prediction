from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def build_pipeline():
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    # Cross-validation report
    cv_scores = cross_val_score(
        grid.best_estimator_,
        X_train,
        y_train,
        cv=cv,
        scoring='f1_macro'
    )

    print("Cross-validation F1-macro scores:", cv_scores)
    print("Mean CV F1-macro:", np.mean(cv_scores))

    return grid.best_estimator_, grid.best_params_
