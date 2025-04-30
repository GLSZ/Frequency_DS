import pandas as pd
from model_training import train_random_forest, split_data

def test_train_random_forest():
    X = pd.DataFrame({"f1": [0, 1, 0, 1], "f2": [1, 0, 1, 0]})
    y = [0, 1, 0, 1]
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.5, random_state=42)
    model = train_random_forest(X_train, y_train)
    assert hasattr(model, "predict")
