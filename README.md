# 50.039 Theory and Practice of Deep Learning

## Setup Instructions

### 1. Git clone this repository

    git clone https://github.com/mavene/2024_TPDL.git

### 2. Download the dataset and model weights

Dataset and Model Weights: [TPDL Project](https://sutdapac-my.sharepoint.com/:f:/g/personal/issac_jose_mymail_sutd_edu_sg/EmFE7PsDCWdHggiCU_82uN4BCrbBrDaG5_-I8D9e23Q42w?e=KCU19s)

Download the directories (‘data/’, ‘models/’ and ‘models-2/’) from the OneDrive Link, TPDL Project and place them in the root directory of the GitHub Repository.

### 3. Download the CheXphoto-v1.0.zip and unzip the file in the root directory of the GitHub Repo.
Upon unzipping, the file directory of the dataset should be as such -
    
    ChexPhoto/
    chexphoto-v1/
    train/
    valid/
    train.csv
    valid.csv

### 4. Download the CheXphoto-valid-v1.1.zip and unzip the file. 

Copy the “valid/” directory and “valid.csv” over to “./ChexPhoto/chexphoto-v1”, replacing the old “valid/” directory and “valid.csv” present in the folder.

### 5. Download the chexlocalize.zip and unzip the file. 

Copy the “test/” directory and “test.csv” over to “./ChexPhoto/chexphoto-v1”. The “chexphoto-v1” directory will now be structured as follows -

    chexphoto-v1/
    test/
    train/
    valid/
    test.csv
    train.csv
    valid.csv

### 6. Setup and activate a Python virtual environment at the root folder of the GitHub Repository. 

Install the required python packages using -

    pip install -r requirements.txt

### 7. Done!

You have now fully set up the environment. 

Navigate to notebooks/ to explore and run the notebooks. Please remember to change the DATA_PATH variable to point to the root directory of the GitHub Repository before running the notebook