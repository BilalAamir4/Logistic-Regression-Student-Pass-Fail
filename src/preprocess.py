import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    return df

def create_target(df):
    df['pass'] = (df['exam_score'] >= 50).astype(int)
    return df

def normalize(X):
    return (X - X.mean()) / X.std()

def preprocess(path):
    df = load_data(path)
    df = create_target(df)

    X = df[['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']]
    y = df['pass']

    X = normalize(X)

    return X.values, y.values