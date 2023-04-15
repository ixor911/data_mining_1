from sklearn import naive_bayes


class Model:
    def __init__(self):
        self.model = naive_bayes.GaussianNB()
        self.name = "naiveBayes"

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train.tolist())

    def predict(self, x_test):
        return self.model.predict(x_test)


