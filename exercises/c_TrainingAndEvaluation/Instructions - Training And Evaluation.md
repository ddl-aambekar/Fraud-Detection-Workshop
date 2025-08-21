# Training And Evaluation
In this phase, we will simultaneously train 3 models, evaluate them, and register the best using a coordinated workflow.  We will execute a Domino Flow that trains the three models, evaluate the models using the Domino Experiment Manager, and register the best ones in the Model Registry.

## Exercise Instructions

In the workspace, open the terminal

Run the Flow

Watch the Flow Execution

Click "Experiment Manager"  (Main Window, Left-Hand Column)

Select all runs (3) and click "Compare"

Select the one with the best metrics (here is a hint: XGBoost)

Review the complete traceability and all factors.

Click "Register Model From Run" in Upper Right Hand

Create model name

In governnace 

This concludes the "3. MODEL TRAINING and EVALUATION" section of the workshop.

## New Domino Concepts

**Domino Flows:**
> Domino Flows is a visual workflow orchestration tool that allows users to build, schedule, and monitor complex data pipelines by connecting multiple tasks, jobs, and models in a directed acyclic graph (DAG). This enables teams to productionize end-to-end ML workflows with built-in error handling, dependency management, and automatic retries without writing orchestration code.

**Experiment Manager:**
> Experiment Manager is a centralized tracking system that automatically captures and compares all experiment runs, including parameters, metrics, code versions, and results in a searchable interface. This accelerates model development by enabling data scientists to quickly identify the best-performing models, understand what changes improved performance, and reproduce any past experiment.

**Model Registry:**
> Model Registry is a centralized repository that catalogs all trained models with their metadata, performance metrics, lineage, and deployment status throughout their lifecycle. This provides governance and collaboration capabilities by enabling teams to discover, compare, promote, and deploy models while maintaining full auditability and compliance documentation.
