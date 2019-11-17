# Disaster Response Pipeline Project

### Table of contents
	1. Description
	2. Dependencies
	3. Installation
	4. Instrauctions to run project
	5. Acknowledgement


### Description:

This project was implemented for the udacity Datascientist nanodegree aiming to classify messages in real time and project was divided into following categories

	1. ETL pipeline to extract data from source and preprocess the data
	2. ML pipeline to train a model that classify text messages in categories
	3. Web application using flask to show results in real time


### Dependencies:

	1. python 3.5+
	2. Tested in linux and Macos


### Installation

	pip install requirements.txt


### Instrauctions to run project:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`



### Acknowledgement:

I thank Udacity for teaching on how to build end to end pipelines using real time data