from sklearn.impute import SimpleImputer
import pandas as pd

NUMERICAL = ['age','trestbps','chol','thalach','oldpeak']
CATEGORICAL = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

def preprocess(df):
    imp_num = SimpleImputer(strategy='median')
    imp_cat = SimpleImputer(strategy='most_frequent')

    df[NUMERICAL] = imp_num.fit_transform(df[NUMERICAL])
    df[CATEGORICAL] = imp_cat.fit_transform(df[CATEGORICAL])

    return df

def merge_classes(x):
    if x == 0:
        return 0
    elif x in [1,2]:
        return 1
    return 2

def encode_features(df):
    # 1. Handle Target (Only if it exists)
    if 'target' in df.columns:
        df['target_multi'] = df['target'].apply(merge_classes)
        y = df['target_multi']
        # Drop both targets for X
        drop_cols = ['target', 'target_multi']
    else:
        y = None
        drop_cols = ['target'] # Only drop if it exists

    # 2. Generate Features (X)
    # errors='ignore' prevents a crash if 'target' isn't there
    X = pd.get_dummies(df.drop(drop_cols, axis=1, errors='ignore'), 
                       columns=[col for col in CATEGORICAL if col in df.columns])
    
    return X, y
