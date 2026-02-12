import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from examples.titanic.functions import train_model
from flexecutor import StageContext
from flexecutor.storage.storage import FlexData
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage


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


if __name__ == "__main__":

    @flexorchestrator(bucket="octavio-flexecutor-bucket")
    def main():
        dag = DAG("titanic")

        stage = Stage(
            stage_id="stage",
            func=train_model,
            inputs=[FlexData(prefix="titanic")],
            outputs=[
                FlexData(
                    prefix="titanic-accuracy",
                    suffix=".txt",
                )
            ],
        )

        dag.add_stage(stage)
        executor = DAGExecutor(dag)
        results = executor.execute(num_workers=3)
        print(results["stage"].get_timings())

    main()
