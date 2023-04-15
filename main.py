import json
import functions
import oneRule
import naiveBayes
import decisionTree
import kNN


features = ['q1', 'q2', 'q3', 'q4']
main_feature = 's'
data = json.load(open("data.json", 'r'))

models = [
    oneRule.Model(),
    naiveBayes.Model(),
    decisionTree.Model(),
    kNN.Model()
]


for key in data.keys():
    dataset = functions.data_convert(data.get(key), features + [main_feature])

    y = dataset[main_feature]
    x = dataset.drop(main_feature, axis=True)

    x_train = x[:10]
    y_train = y[:10]
    x_test = x[10:]
    # y_test = y[10:]

    print(f"{key}:")
    for model in models:
        model.train(x_train, y_train)
        predict = model.predict(x_test)
        print(f"\t{model.name}: {predict[0]}")










