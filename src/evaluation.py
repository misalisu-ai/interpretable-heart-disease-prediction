from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    return y_pred, cm
