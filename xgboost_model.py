import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score



def read_trainset(file_path):
    data = pd.read_excel(file_path).values
    print(data.shape)
    labels = data[:, 0]
    labels = labels.astype(np.uint8)
    inputs = data[:, 1:]

    return inputs, labels


def read_testset(file_path):
    data = pd.read_csv(file_path).values
    print(data.shape)
    labels = data[:, 0]
    labels = labels.astype(np.uint8)
    inputs = data[:, 1:]

    return inputs


def train(trainset, testset=None):
    inputs, labels = trainset
    print(f"The size of inputs: {inputs.shape}")
    print(f"The size of labels: {labels.shape}")
    
    validset = [(inputs, labels)]

    xgbclassifier = XGBClassifier(objective="binary:logistic")
    # xgbclassifier.fit(inputs, labels, eval_metric="logloss", eval_set=validset)
    xgbclassifier.fit(inputs, labels)

    preds = xgbclassifier.predict(inputs)
    preds = preds >= 0.5
    preds = preds.astype(np.uint8)
    accuracy = accuracy_score(labels, preds)
    print(f"The accuracy on trainset: {accuracy}")


    testset_classes = xgbclassifier.predict(testset)
    testset_probs = xgbclassifier.predict_proba(testset)

    print(testset_probs.shape)
    with open("./outputs/xgboost.txt", "w", encoding="utf-8") as f:
        f.write("object, prob, class\n")
        max_index = testset_probs.shape[0]
        for index in range(max_index):
            prob = testset_probs[index, testset_classes[index]]
            category = testset_classes[index]
            f.write(f"{index+1},{prob}, {category}\n")


    




if __name__ == "__main__":
    trainset_path = "./dataset/雪崩数据库308.xlsx"
    testset_path = "./dataset/fishnet.txt"
    trainset = read_trainset(trainset_path)
    testset = read_testset(testset_path)
    model = train(trainset, testset)



