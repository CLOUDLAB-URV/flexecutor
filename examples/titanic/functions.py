import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from flexecutor import StageContext


def train_model(ctx: StageContext) -> None:
    chunk_path = ctx.get_input_paths("titanic")[0]
    chunk = pd.read_csv(chunk_path)
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

    with open(ctx.next_output_path("titanic-accuracy"), "w") as f:
        f.write(str(accuracy))
