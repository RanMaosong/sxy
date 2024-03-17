import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score



def read_trainset(file_path):
    data = pd.read_excel(file_path).values
    print(data.shape)
    labels = data[:, 0]
    labels = labels.astype(np.uint8)
    inputs = data[:, 1:]

    return inputs, labels

def set_model_config():
    params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'gamma': 0.1,
    'max_depth': 8,
    'alpha': 0,
    'lambda': 0,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'min_child_weight': 3,
    'silent': 0,
    'eta': 0.03,
    'nthread': -1,
    'seed': 2019,
}

    return params

def train(inputs, labels):
    print(f"The size of inputs: {inputs.shape}")
    print(f"The size of labels: {labels.shape}")
    
    validset = [(inputs, labels)]

    xgbclassifier = XGBClassifier(objective="binary:logistic")
    xgbclassifier.fit(inputs, labels, eval_metric="logloss", eval_set=validset)

    preds = xgbclassifier.predict(inputs)
    preds = preds >= 0.5
    preds = preds.astype(np.uint8)
    accuracy = np.mean(preds)

    print(f"The accuracy on trainset: {accuracy}")
    accuracy = accuracy_score(labels, preds)
    print(f"The accuracy on trainset: {accuracy}")




if __name__ == "__main__":
    trainset_path = "./雪崩数据库308.xlsx"
    inputs, labels = read_trainset(trainset_path)
    train(inputs, labels)



