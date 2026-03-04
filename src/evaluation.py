from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate(model, X_train, y_train, X_test, y_test, cv_folds=5):
    """
    Produces the Classification Table PLUS the Cross-Validation Stability Report.
    """
    y_pred = model.predict(X_test)

    print("\n--- Test Set Classification Report ---")
    print(classification_report(y_test, y_pred))

    print(f"\n--- {cv_folds}-Fold Cross-Validation Report ---")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
    
    print(f"Individual Fold Accuracies: {np.round(cv_scores, 3)}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    return y_pred, cm, cv_scores
