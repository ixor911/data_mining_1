import pandas as pd


class Model:
    def __init__(self):
        self.most_common_class = None
        self.name = "oneRule"

    def train(self, x_train, y_train):
        class_counts = y_train.value_counts()
        self.most_common_class = class_counts.idxmax()

    def predict(self, x_test):
        predict = []
        for i in range(0, len(x_test)):
            predict.append(self.most_common_class)

        return predict
