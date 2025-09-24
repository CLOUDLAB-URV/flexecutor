from lithops import Storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd
import lithops
import time
import os

num_workers = 3
bucket = "octavio-flexecutor-bucket"
prefix = "titanic/"


def train_model(keys) -> None:
    storage = Storage()

    read_init = time.time()

    paths = []
    os.makedirs(f"/tmp/{prefix}", exist_ok=True)
    for item in keys:
        path = "/tmp/" + item
        paths.append(path)
        storage.download_file(bucket, item, path)

    read_end = time.time()

    chunk = pd.read_csv(paths[0])
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    chunk = chunk.dropna(subset=features + ["Survived"])

    X = chunk[features]
    X = pd.get_dummies(X, columns=["Sex"], drop_first=True)
    y = chunk["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    write_start = time.time()

    storage.put_object(bucket, prefix + "-accuracy.txt", str(accuracy))

    write_end = time.time()
    read = read_end - read_init
    write = write_end - write_start
    compute = write_start - read_end

    return (read, compute, write)


if __name__ == "__main__":

    def main():
        fexec = lithops.FunctionExecutor()

        # explicit scatter the files
        storage = lithops.Storage()
        objects = storage.list_objects(bucket, prefix=prefix)
        keys = [obj["Key"] for obj in objects if obj["Key"][-1] != "/"]

        # split keys in list of length num_workers (no def fucntion)
        iterdata = [keys[i: i + len(keys)//num_workers]
                    for i in range(0, len(keys), len(keys)//num_workers)]
        fexec.map(train_model, iterdata)
        fexec.wait()
        profilings = fexec.get_result()
        print(profilings)

        return 0

    main()
