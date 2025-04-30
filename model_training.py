from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    return clf
