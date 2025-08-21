# Fraud-Detection-Workshop
This workshop is designed to provide attendees with a hands-on experience experience with the Domino Data Lab platform while completing wortking through the complete model development and delivery lifecycle.  It takes the user from data preparation through to model delivery.

## Exercises
Below are the exercises attendees will complete as part of the workshop.  The exercise directories correspond with each exercise and include all resources needed to complete the exercise including instructions, slides, notebooks, and scripts.  Each exercise builds upon the other.  As such they must be completed in sequential order:

### 1 - Data Exploration 
Interactive notebook format. Read data from s3 to dataframe. Remove dirty data. Generate plots. Save to Domino Dataset. 

### 2 - Data Engineering
Python script executed as job. Read cleaned Domino Dataset to dataframe. Add derived columns. Normalize/scale/encode cleaned dataframe. Save dataframe to Domino Dataset and snapshot (tbd). Save Scaling/Encoder/Normalization model.

### 3 - Model Training & Evaluation
Job executing 3 different training scripts and saving results to Experiments Manager. Add fairness/bias evaluation.

Manually register model of choice from experiments page.

### 4 - Delivery & Hosting
- Hosted REST API's
- Hosted Web Apps
- Launchers