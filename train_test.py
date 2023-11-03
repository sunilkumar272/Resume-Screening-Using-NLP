def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
def evaluate_model(model, X, y):
    accuracy=model.score(X, y)
    return accuracy