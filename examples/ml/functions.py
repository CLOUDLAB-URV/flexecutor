import multiprocessing as mp

import lightgbm as lgb
import numpy as np
from joblib import dump, load
from numpy.linalg import eig
from sklearn.base import BaseEstimator

from flexecutor import StageContext


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


def pca(ctx: StageContext):
    [training_data_path] = ctx.get_input_paths("training-data")

    train_data = np.genfromtxt(training_data_path, delimiter="\t")
    train_labels = train_data[:, 0]
    a = train_data[:, 1 : train_data.shape[1]]
    ma = np.mean(a.T, axis=1)
    ca = a - ma
    va = np.cov(ca.T)
    values, vectors = eig(va)
    pa = vectors.T.dot(ca.T)

    vectors_pca_path = ctx.next_output_path("vectors-pca")
    training_data_transform = ctx.next_output_path("training-data-transform")
    np.savetxt(vectors_pca_path, vectors, delimiter="\t")
    first_n_a = pa.T[:, 0:100].real
    train_labels = train_labels.reshape(train_labels.shape[0], 1)
    first_n_a_label = np.concatenate((train_labels, first_n_a), axis=1)
    np.savetxt(training_data_transform, first_n_a_label, delimiter="\t")


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
    # accuracy = calc_accuracy(y_pred, y_train)

    return gbm


def train_with_multiprocessing(ctx: StageContext):
    # TODO: make that number of processes launched in training can be defined by user
    task_id = 0
    num_process = 12
    param = {"feature_fraction": 1, "max_depth": 8, "num_of_trees": 30, "chance": 1}

    [training_data_path] = ctx.get_input_paths("training-data-transform")

    with mp.Pool(processes=num_process) as pool:
        results = pool.starmap(
            train,
            [
                (
                    ctx,
                    task_id,
                    i,
                    param["feature_fraction"],
                    param["max_depth"],
                    param["num_of_trees"],
                    param["chance"],
                    training_data_path,
                )
                for i in range(num_process)
            ],
        )
        for result in results:
            model_path = ctx.next_output_path("models")
            result.save_model(model_path)


def calc_accuracy(y_pred, y_train):
    count_match = 0
    for i in range(len(y_pred)):
        result = np.where(y_pred[i] == np.amax(y_pred[i]))[0]
        if result == y_train[i]:
            count_match = count_match + 1
    # The accuracy on the training set
    accuracy = count_match / len(y_pred)
    return accuracy


def aggregate(ctx: StageContext):
    [training_data_path] = ctx.get_input_paths("training-data-transform")
    model_paths = ctx.get_input_paths("models")

    test_data = np.genfromtxt(training_data_path, delimiter="\t")
    y_test = test_data[5000:, 0]
    x_test = test_data[5000:, 1 : test_data.shape[1]]
    model_list = []

    for model_path in model_paths:
        model = lgb.Booster(model_file=model_path)
        model_list.append(model)

    # Merge models
    forest = MergedLGBMClassifier(model_list)
    forest_path = ctx.next_output_path("forests")
    forest.save_model(forest_path)

    # Predict
    y_pred = forest.predict(x_test)
    acc = calc_accuracy(y_pred, y_test)
    prediction_path = ctx.next_output_path("predictions")
    np.savetxt(prediction_path, y_pred, delimiter="\t")

    return acc


def test(ctx: StageContext):
    predictions_paths = ctx.get_input_paths("predictions")
    predictions = [
        np.genfromtxt(prediction_path, delimiter="\t")
        for prediction_path in predictions_paths
    ]
    [test_path] = ctx.get_input_paths("training-data-transform")
    test_data = np.genfromtxt(test_path, delimiter="\t")

    y_test = test_data[5000:, 0]
    y_pred = sum(predictions) / len(predictions)
    acc = calc_accuracy(y_pred, y_test)

    accuracy_path = ctx.next_output_path("accuracies")
    with open(accuracy_path, "w") as f:
        f.write("My accuracy is: " + str(acc))
