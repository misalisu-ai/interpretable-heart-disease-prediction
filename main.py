from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import preprocess, encode_features
from src.feature_engineering import scale_data
from src.model import build_pipeline, tune_model
from src.evaluation import evaluate

# Load
df = load_data()

# Preprocess
df = preprocess(df)
X, y = encode_features(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

# Model
pipe = build_pipeline()
best_model, params = tune_model(pipe, X_train_scaled, y_train)

print("Best parameters:", params)

# Evaluate
evaluate(best_model, X_test_scaled, y_test)
