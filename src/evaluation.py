from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate(model, X, y, X_test, y_test, cv_folds=5):
    """
    Performs both a specific test-set evaluation and a 
    generalized Cross-Validation report.
    """
    y_pred = model.predict(X_test)
    
    print("--- Single Test Set Evaluation ---")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cv_scores = cross_val_score(model, X, y, cv=cv_folds)
    
    print(f"\n--- {cv_folds}-Fold Cross-Validation ---")
    print(f"Scores: {np.round(cv_scores, 3)}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    return y_pred, cm, cv_scores
