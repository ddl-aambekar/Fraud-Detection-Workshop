import os
from flytekit import workflow
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask

# pyflyte run --remote --name train-models exercises/d_TrainingAndEvaluation/workflow.py credit_card_fraud_detection_workflow 
DOMINO_WORKING_DIR = os.environ["DOMINO_WORKING_DIR"]

# --- Task to provide the transformed file path ---
provide_transformed_file = DominoJobTask(
    name="provide-transformed-file",
    domino_job_config=DominoJobConfig(
        # Writes the absolute path to the transformed CSV
        Command=(
            'bash -c "echo /mnt/data/Fraud-Detection-Workshop/transformed_cc_transactions.csv > transformed_filename"'
        )
    ),
    inputs={},
    outputs={"transformed_filename": str},
    use_latest=True,
    cache=False,
)

# --- Domino job tasks for training ---
ada_training_task = DominoJobTask(
    name="train-ada",
    domino_job_config=DominoJobConfig(
        Command=f"python {DOMINO_WORKING_DIR}/exercises/d_TrainingAndEvaluation/trainer_ada.py"
    ),
    inputs={"transformed_filename": str},
    outputs={"results": str},
    use_latest=True,
    cache=True,
)

gnb_training_task = DominoJobTask(
    name="train-gnb",
    domino_job_config=DominoJobConfig(
        Command=f"python {DOMINO_WORKING_DIR}/exercises/d_TrainingAndEvaluation/trainer_gnb.py"
    ),
    inputs={"transformed_filename": str},
    outputs={"results": str},
    use_latest=True,
    cache=True,
)

xgb_training_task = DominoJobTask(
    name="train-xgb",
    domino_job_config=DominoJobConfig(
        Command=f"python {DOMINO_WORKING_DIR}/exercises/d_TrainingAndEvaluation/trainer_xgb.py"
    ),
    inputs={"transformed_filename": str},
    outputs={"results": str},
    use_latest=True,
    cache=True,
)

# --- Compare results task ---
compare_task = DominoJobTask(
    name="compare-results",
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
    # Step 1: Provide the transformed CSV path
    transformed_filename = provide_transformed_file()

    # Step 2: Train models
    ada_results = ada_training_task(transformed_filename=transformed_filename)
    gnb_results = gnb_training_task(transformed_filename=transformed_filename)
    xgb_results = xgb_training_task(transformed_filename=transformed_filename)

    # Step 3: Compare results
    comparison = compare_task(
        ada_results=ada_results,
        gnb_results=gnb_results,
        xgb_results=xgb_results
    )

    return comparison
