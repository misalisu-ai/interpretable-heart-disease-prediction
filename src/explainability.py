import shap

def explain_model(model, X_train, X_test, feature_names):
    X_train_sel = model.named_steps['rfe'].transform(X_train)
    X_test_sel = model.named_steps['rfe'].transform(X_test)

    explainer = shap.Explainer(
        model.named_steps['logreg'],
        X_train_sel,
        feature_names=feature_names
    )

    shap_values = explainer(X_test_sel)

    shap.summary_plot(
        shap_values,
        X_test_sel,
        feature_names=feature_names
    )
