from lithops import FunctionExecutor

from examples.ml.functions import pca, train_with_multiprocessing, aggregate, test
from flexecutor.utils.utils import flexorchestrator


from joblib import Parallel, delayed

import lightgbm as lgb
import numpy as np
from joblib import dump, load
from numpy.linalg import eig
from sklearn.base import BaseEstimator

num_workers = 4
bucket = "octavio-flexecutor-bucket"


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


def pca(training_keys):

    storage = Storage()
    read_init = time.time()

    paths = []
    os.makedirs("/tmp/training-data", exist_ok=True)
    for item in training_keys:
        path = "/tmp/" + item
        paths.append(path)
        storage.download_file(bucket, item, path)

    read_end = time.time()

    train_data = np.genfromtxt(paths[0], delimiter="\t")
    train_labels = train_data[:, 0]
    a = train_data[:, 1 : train_data.shape[1]]
    ma = np.mean(a.T, axis=1)
    ca = a - ma
    va = np.cov(ca.T)
    values, vectors = eig(va)
    pa = vectors.T.dot(ca.T)

    vectors_pca_path =f"{'/tmp/vectors_pca_' + uuid.uuid4()[0:8]}.mp4"
    training_data_transform = f"{'/tmp/training_data_transform_' + uuid.uuid4()[0:8]}.mp4"
    np.savetxt(vectors_pca_path, vectors, delimiter="\t")
    first_n_a = pa.T[:, 0:100].real
    train_labels = train_labels.reshape(train_labels.shape[0], 1)
    first_n_a_label = np.concatenate((train_labels, first_n_a), axis=1)
    np.savetxt(training_data_transform, first_n_a_label, delimiter="\t")

    write_start = time.time()
    storage.upload_file(vectors_pca_path, bucket, "vectors-pca/" + os.path.basename(vectors_pca_path))
    storage.upload_file(training_data_transform, bucket, "training-data-transform/" + os.path.basename(training_data_transform))
    write_end = time.time()

    read = read_end - read_init
    write = write_end - write_start
    compute = write_start - read_end

    return (read, compute, write)


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


def train_with_multiprocessing(training_data_keys):
    task_id = 0
    num_process = 1
    param = {"feature_fraction": 1, "max_depth": 8, "num_of_trees": 30, "chance": 1}

    storage = Storage()
    read_init = time.time()

    paths = []
    os.makedirs("/tmp/training-data-transform", exist_ok=True)
    for item in training_data_keys:
        path = "/tmp/" + item
        paths.append(path)
        storage.download_file(bucket, item, path)
    training_data_path = paths[0]

    read_end = time.time()

    results = Parallel(n_jobs=num_process, backend="threading")(
        delayed(train)(
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
    )

    write_start = time.time()
    for result in results:
        model_path = f"{'/tmp/model_' + uuid.uuid4()[0:8]}.txt"
        result.save_model(model_path)
        storage.upload_file(model_path, bucket, "models/" + os.path.basename(model_path))
    write_end = time.time()

    read = read_end - read_init
    write = write_end - write_start
    compute = write_start - read_end

    return (read, compute, write)


def calc_accuracy(y_pred, y_train):
    count_match = 0
    for i in range(len(y_pred)):
        result = np.where(y_pred[i] == np.amax(y_pred[i]))[0]
        if result == y_train[i]:
            count_match = count_match + 1
    # The accuracy on the training set
    accuracy = count_match / len(y_pred)
    return accuracy


def aggregate(transform_keys, model_keys):
    storage = Storage()

    read_init = time.time()

    data_paths = []
    os.makedirs("/tmp/training-data-transform", exist_ok=True)
    for item in transform_keys:
        path = "/tmp/" + item
        data_paths.append(path)
        storage.download_file(bucket, item, path)
    training_data_path = data_paths[0]
    model_paths = []
    os.makedirs("/tmp/models", exist_ok=True)
    for item in model_keys:
        path = "/tmp/" + item
        model_paths.append(path)
        storage.download_file(bucket, item, path)

    read_end = time.time()

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

    write_start = time.time()
    prediction_path = f"{'/tmp/predictions_' + uuid.uuid4()[0:8]}.txt"
    np.savetxt(prediction_path, y_pred, delimiter="\t")
    storage.upload_file(prediction_path, bucket, "predictions/" + os.path.basename(prediction_path))
    write_end = time.time()

    read = read_end - read_init
    write = write_end - write_start
    compute = write_start - read_end

    return (read, compute, write)

def test(prediction_keys, transform_keys):
    storage = Storage()
    read_init = time.time()

    data_paths = []
    os.makedirs("/tmp/predictions", exist_ok=True)
    for item in keys:
        path = "/tmp/" + item
        data_paths.append(path)
        storage.download_file(bucket, item, path)

    predictions = [
        np.genfromtxt(prediction_path, delimiter="\t")
        for prediction_path in data_paths
    ]
    transform_paths = []
    os.makedirs("/tmp/training-data-transform", exist_ok=True)
    for item in transform_keys:
        path = "/tmp/" + item
        transform_paths.append(path)
        storage.download_file(bucket, item, path)
    test_path = transform_paths[0]
    read_end = time.time()

    test_data = np.genfromtxt(test_path, delimiter="\t")

    y_test = test_data[5000:, 0]
    y_pred = sum(predictions) / len(predictions)
    acc = calc_accuracy(y_pred, y_test)

    write_start = time.time()
    accuracy_path = f"{'/tmp/accuracy_' + uuid.uuid4()[0:8]}.txt"
    with open(accuracy_path, "w") as f:
        f.write("My accuracy is: " + str(acc))
    storage.upload_file(accuracy_path, bucket, "accuracies/" + os.path.basename(accuracy_path))
    write_end = time.time()

    read = read_end - read_init
    write = write_end - write_start
    compute = write_start - read_end

    return (read, compute, write)


def explicit_scatter(prefix):
    # explicit scatter the files
    objects = storage.list_objects(bucket, prefix=prefix)
    keys = [obj["Key"] for obj in objects if obj["Key"][-1] != "/"]

    # split keys in list of length num_workers (no def fucntion)
    iterdata = [keys[i: i + len(keys)//num_workers]
                for i in range(0, len(keys), len(keys)//num_workers)]

    return iterdata


if __name__ == "__main__":

    @flexorchestrator(bucket="test-bucket")
    def main():
        fexec = FunctionExecutor()

        training_keys = explicit_scatter("data_training/")
        fexec.map(pca, training_keys)
        fexec.wait()
        profilings = fexec.get_result()

        transform_keys = explicit_scatter("data_training_transform/")
        fexec.map(train_with_multiprocessing, transform_keys)
        fexec.wait()
        profilings = fexec.get_result()

        model_keys = explicit_scatter("models/")
        fexec.map(aggregate, transform_keys, model_keys)
        fexec.wait()
        profilings = fexec.get_result()

        test_keys = explicit_scatter("predictions/")
        fexec.map(test, test_keys, transform_keys)
        fexec.wait()
        profilings = fexec.get_result()

    main()
