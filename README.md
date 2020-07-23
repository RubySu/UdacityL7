# Sparkify Users Churn Analysis

### Project Overview  

Sparkify is a music app, this project is based on the user interactions of log data to have user churn analysis.
In this project, we are going to identify the users who has high possibility to leave Sparkify. I used a tiny subset (128MB) of the full dataset (12GB) with pyspark.ml to build classification model.

### Problem Statement

##### What is the definition of Churn?
Using the `Cancellation Confirmation` events to define your churn, which happen for both paid and free users. O means non churn group, 1 means churn group.
##### What is the distribution of free and paid users between churn group and non churn group?
We cannot find significant difference in between.  

##### What is the distribution by different page types between churn group and non churn group?
From page information, we can find churn users did not like access 'Thumbs up','Home','Add to Playlist','Next Song' usually. 

##### What is played number of songs per user between churn group and non churn group?
We could find the churn group only played less than 18 songs usually, non churn group played more than 30 songs usually.

##### How to predict the churn users? What is accuracy of the model on test set?
We used RandomForest,LogisticRegression,DecisionTree as classifiers, finally LogisticRegression had been chosen according to highest F1-score, so far the accuracy is 0.70 on test data set.

### Jupyter

For detail analysis, please refer to Sparkify.ipynb.

### Flask Web App

A a Flask web application has been integrated in this project, please follow the instructions below to start a local instance.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run data pipeline that cleans data with feature engineering and persist data.
        `python process_data.py mini_sparkify_event_data.json level_churn_byUser.csv page_churn_byUser.csv numofsongs_byUser.csv user_item.csv`
    - To run ML pipeline that trains classifier and saves
        `python train_classifier.py ../data/user_item.csv rf lr dt`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

- app
| - template
| |- master.html  # main page of web app with distribution of churn users by level
| |- report.html  # report file for distribution of churn users by page
|- run.py  # Flask file that runs app

- data
|- f_coef.csv  # prediction result evaluation 
|- level_churn_byUser.csv  # subset of churn users by level for visualization
|- page_churn_byUser.csv # subset of churn users by page for visualization
|- mini_sparkify_event_data.json # data to be process
|- process_data.py # data process file
|- user_item.csv   # user item matrix after data process

- models
|- rf # saved RandomForest model
|- lr # saved LinearRegression model
|- dt # saved DecisionTree model
|- model_training.py  # train and save models
