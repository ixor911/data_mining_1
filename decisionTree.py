from sklearn import tree


class Model:
    def __init__(self):
        self.model = tree.DecisionTreeClassifier()
        self.name = "decisionTree"

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train.tolist())

    def predict(self, x_test):
        return self.model.predict(x_test)