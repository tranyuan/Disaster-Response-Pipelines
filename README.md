# Disaster Response Pipeline Project

### Introduction:

In the Project, the data set containing real messages that were sent during disaster events. I have created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

The project include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Packages:

1. Machine Learning : Pandas, Sciki-Learn
2. Natural Language Process : NLTK
3. SQLlite Database : SQLalchemy
4. Web App and Data Visualization: Flask, Plotly

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
