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
    df['target_multi'] = df['target'].apply(merge_classes)
    X = pd.get_dummies(df.drop(['target','target_multi'], axis=1),
                       columns=CATEGORICAL)
    y = df['target_multi']
    return X, y
