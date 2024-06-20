import multiprocessing as mp
import os
from random import random
from flexecutor.utils.utils import IOManager

from numpy.linalg import eig
from sklearn.base import BaseEstimator
from joblib import dump, load
import numpy as np
import lightgbm as lgb


class MergedLGBMClassifier(BaseEstimator):
    def __init__(self, model_list):
        assert isinstance(model_list, list)

        self.model_list = model_list

    def predict(self, X):
        pred_list = []

        for m in self.model_list:
            pred_list.append(m.predict(X))

        # Average the predictions
        averaged_preds = sum(pred_list) / len(pred_list)

        return averaged_preds

    def save_model(self, model_path):
        dump(self, model_path)

    @staticmethod
    def load_model(model_path):
        return load(model_path)


def pca(io: IOManager):
    [training_data_path] = io.input_paths("training-data")

    train_data = np.genfromtxt(training_data_path, delimiter="\t")
    train_labels = train_data[:, 0]
    a = train_data[:, 1 : train_data.shape[1]]
    ma = np.mean(a.T, axis=1)
    ca = a - ma
    va = np.cov(ca.T)
    values, vectors = eig(va)
    pa = vectors.T.dot(ca.T)

    vectors_pca_path = io.output_paths("vectors-pca")
    training_data_transform = io.output_paths("training-data-transform")
    np.savetxt(vectors_pca_path, vectors, delimiter="\t")
    first_n_a = pa.T[:, 0:100].real
    train_labels = train_labels.reshape(train_labels.shape[0], 1)
    first_n_a_label = np.concatenate((train_labels, first_n_a), axis=1)
    np.savetxt(training_data_transform, first_n_a_label, delimiter="\t")


def train_with_multiprocessing(io: IOManager):
    task_id = 0
    num_process = 48
    num_vcpu = 6

    processes = []
    manager = mp.Manager()
    res_dict = manager.dict()
    param = {"feature_fraction": 1, "max_depth": 8, "num_of_trees": 30, "chance": 1}

    training_data_path = io.input_paths("training-data-transform")

    for i in range(num_process):
        feature_fraction = round(random() / 2 + 0.5, 1)
        chance = round(random() / 2 + 0.5, 1)

        param["feature_fraction"] = feature_fraction
        param["chance"] = chance

        pro = mp.Process(
            target=train,
            args=(
                io,
                task_id,
                i,
                param["feature_fraction"],
                param["max_depth"],
                param["num_of_trees"],
                param["chance"],
                training_data_path,
            ),
        )
        processes.append(pro)

    start_ids = 0
    end_ids = start_ids + num_vcpu

    while start_ids < num_process:
        end_ids = min(end_ids, num_process)
        for i in range(start_ids, end_ids):
            processes[i].start()

        for i in range(start_ids, end_ids):
            processes[i].join()

        start_ids += num_vcpu
        end_ids += num_vcpu

    return None


def train(
    io,
    task_id,
    process_id,
    feature_fraction,
    max_depth,
    num_of_trees,
    chance,
    training_path,
):
    train_data = np.genfromtxt(training_path, delimiter="\t")
    y_train = train_data[0:5000, 0]
    x_train = train_data[0:5000, 1 : train_data.shape[1]]

    _id = str(task_id) + "_" + str(process_id)
    params = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "num_classes": 10,
        "metric": {"multi_logloss"},
        "num_leaves": 50,
        "learning_rate": 0.05,
        "feature_fraction": feature_fraction,
        "bagging_fraction": chance,  # If model indexes are 1->20, this makes feature_fraction: 0.7->0.9
        "bagging_freq": 5,
        "max_depth": max_depth,
        "verbose": -1,
        "num_threads": 2,
    }

    lgb_train = lgb.Dataset(x_train, y_train)
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=num_of_trees,
        valid_sets=lgb_train,
        # early_stopping_rounds=5
    )

    y_pred = gbm.predict(x_train, num_iteration=gbm.best_iteration)
    accuracy = calc_accuracy(y_pred, y_train)

    model_path = io.next_output_path("model-path")
    gbm.save_model(model_path)

    return accuracy


def calc_accuracy(y_pred, y_train):
    count_match = 0
    for i in range(len(y_pred)):
        result = np.where(y_pred[i] == np.amax(y_pred[i]))[0]
        if result == y_train[i]:
            count_match = count_match + 1
    # The accuracy on the training set
    accuracy = count_match / len(y_pred)
    return accuracy


def aggregate(io: IOManager):
    [training_data_path] = io.input_paths("training-data-transform")
    model_paths = io.input_paths("models")

    test_data = np.genfromtxt(training_data_path, delimiter="\t")
    y_test = test_data[5000:, 0]
    x_test = test_data[5000:, 1 : test_data.shape[1]]
    model_list = []

    for model_path in model_paths:
        model = lgb.Booster(model_file=model_path)
        model_list.append(model)

    # Merge models
    forest = MergedLGBMClassifier(model_list)
    [forest_path] = io.output_paths("forests")
    forest.save_model(forest_path)

    # Predict
    y_pred = forest.predict(x_test)
    acc = calc_accuracy(y_pred, y_test)
    [prediction_path] = io.output_paths("predictions")
    np.savetxt(prediction_path, y_pred, delimiter="\t")

    return acc


def test(io: IOManager):
    predictions_paths = io.input_paths("predictions")
    predictions = [
        np.genfromtxt(prediction_path, delimiter="\t")
        for prediction_path in predictions_paths
    ]
    test_path = io.input_paths("test")
    test_data = np.genfromtxt(test_path, delimiter="\t")

    y_test = test_data[5000:, 0]
    y_pred = sum(predictions) / len(predictions)
    acc = calc_accuracy(y_pred, y_test)

    return acc
