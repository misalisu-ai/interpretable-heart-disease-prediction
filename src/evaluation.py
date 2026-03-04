from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate(model, X_train, y_train, X_test, y_test):
    """
    Performs Test-set Evaluation AND Cross-Validation for Research.
    """
    y_pred = model.predict(X_test)
    
    print("\n--- Test Set Classification Report ---")
    print(classification_report(y_test, y_pred))

    print("\n--- 5-Fold Cross-Validation Report ---")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    print(f"Fold Accuracies: {np.round(cv_scores, 3)}")
    print(f"Mean Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    return y_pred, cm
