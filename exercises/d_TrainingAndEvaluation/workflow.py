import os
from flytekit import workflow, task, FlyteFile
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask

DOMINO_WORKING_DIR = os.environ["DOMINO_WORKING_DIR"]
DOMINO_DATASETS_DIR = os.environ["DOMINO_DATASETS_DIR"]

# --- Task to "provide" the transformed CSV file ---
# If the file is already in /mnt/data in the first job, we just wrap it as a FlyteFile output.
@task
def provide_transformed_file() -> FlyteFile:
    # Path inside this job to the transformed file
    path = f"{DOMINO_DATASETS_DIR}/Fraud-Detection-Workshop/transformed_cc_transactions.csv"
    return FlyteFile(path)


# --- Domino job tasks for training ---
ada_training_task = DominoJobTask(
    name="Train AdaBoost classifier",
    domino_job_config=DominoJobConfig(
        Command=f"python {DOMINO_WORKING_DIR}/exercises/d_TrainingAndEvaluation/trainer_ada.py"
    ),
    inputs={"transformed_file": FlyteFile},
    outputs={"results": str},
    use_latest=True,
    cache=True,
)

gnb_training_task = DominoJobTask(
    name="Train GaussianNB classifier",
    domino_job_config=DominoJobConfig(
        Command=f"python {DOMINO_WORKING_DIR}/exercises/d_TrainingAndEvaluation/trainer_gnb.py"
    ),
    inputs={"transformed_file": FlyteFile},
    outputs={"results": str},
    use_latest=True,
    cache=True,
)

xgb_training_task = DominoJobTask(
    name="Train XGBoost classifier",
    domino_job_config=DominoJobConfig(
        Command=f"python {DOMINO_WORKING_DIR}/exercises/d_TrainingAndEvaluation/trainer_xgb.py"
    ),
    inputs={"transformed_file": FlyteFile},
    outputs={"results": str},
    use_latest=True,
    cache=True,
)

# --- Compare results task ---
compare_task = DominoJobTask(
    name="Compare training results",
    domino_job_config=DominoJobConfig(
        Command=f"python {DOMINO_WORKING_DIR}/exercises/d_TrainingAndEvaluation/compare.py"
    ),
    inputs={"ada_results": str, "gnb_results": str, "xgb_results": str},
    outputs={"consolidated": str},
    use_latest=True,
)

# --- Workflow definition ---
@workflow
def credit_card_fraud_detection_workflow() -> str:
    # Step 1: Produce or locate the transformed CSV
    transformed_file = provide_transformed_file()

    # Step 2: Train models (Flyte passes file between jobs)
    ada_results = ada_training_task(transformed_file=transformed_file)
    gnb_results = gnb_training_task(transformed_file=transformed_file)
    xgb_results = xgb_training_task(transformed_file=transformed_file)

    # Step 3: Compare results
    comparison = compare_task(
        ada_results=ada_results,
        gnb_results=gnb_results,
        xgb_results=xgb_results
    )

    return comparison
