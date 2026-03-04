import pandas as pd

COLUMNS = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal','target'
]

def load_data(path="data/cleveland.csv"):
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values='?')
    return df
