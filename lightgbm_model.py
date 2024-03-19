import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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

def train(trainset, testset, testset_output_path):
    inputs, labels = trainset
    print(f"The size of inputs: {inputs.shape}")
    print(f"The size of labels: {labels.shape}")
    
    validset = [(inputs, labels)]

    lgbmclassifier = LGBMClassifier()
    lgbmclassifier.fit(inputs, labels, eval_metric="logloss", eval_set=validset)

    preds = lgbmclassifier.predict(inputs)
    preds = preds >= 0.5
    preds = preds.astype(np.uint8)
    accuracy = accuracy_score(labels, preds)
    print(f"The accuracy on trainset: {accuracy}")

    testset_classes = lgbmclassifier.predict(testset)
    testset_probs = lgbmclassifier.predict_proba(testset)

    print(testset_probs.shape)
    with open(testset_output_path, "w", encoding="utf-8") as f:
        f.write("object, prob, class\n")
        max_index = testset_probs.shape[0]
        for index in range(max_index):
            prob = testset_probs[index, testset_classes[index]]
            category = testset_classes[index]
            f.write(f"{index+1},{prob}, {category}\n")


if __name__ == "__main__":
    trainset_path = "./dataset/雪崩数据库308.xlsx"
    testset_path = "./dataset/fishnet.txt"
    testset_output_path = "./outputs/lightgbm.txt"
    trainset = read_trainset(trainset_path)
    testset = read_testset(testset_path)
    model = train(trainset, testset, testset_output_path)



