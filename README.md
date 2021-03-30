# Disaster Response Message Classification

## Description

 In the times of disasters, messages from public needs to be directed to the concerned departments for quick and 
 responsive help. With the help of NLP and Machine learning, messages a sorted and directed by a trained Machine 
 Learning model.

## Dependencies

- Python 3.5+
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database: SQLalchemy
- Machine Learning Model Saving and Loading Library: Pickle
- Web deployment and Data Visualization: Flask, Plotly

## Workflow Description

### ETL (Extract, Transform and Load)

- Data is acquired from two csv files disaster_categories.csv and disaster_messages.csv

1. **Extracting**: 
   Using Pandas library, the two csv files are read into the two dataframes. The two dataframes are
   merged together
   
2. **Transforming**: 
   Data is cleaned by creating new target columns based on the type of message as labeled by the 
   database collector. Also, dropping the duplicated observations/rows. After this step, the data used for creating the 
   model is ready for modelling preprocessing and processing.
   
3. **Loading/Saving**: 
   The prepared data is saved, and later loaded, by the use of SQLite.

### Modelling

- In this step, the data is preprocessed from plain text into a data structure ready to be learnt by a machine 
  learning model, training the model and Saving a trained model file to be used in later prediction of the 
  type of message hence the respective department.
  
#### 1. **Preprocessing**: 
   Preprocessing a plain text into numerical representation that are suitable for machine learning model is done through
   two steps:
   
   ###### 1.1 Vectorization: 

   Plain text are turned into vectors based on the count of words extracted from the text
   with the following characteristics: low casing, lemmatized and unique/specialized words.

   ###### 1.2 TF-IDF transformer: 

   With Tfidftransformer the preprocessor will systematically compute word counts using CountVectorizer and 
   then compute the Inverse Document Frequency (IDF) values and only then compute the Tf-idf scores. Which would place 
   great emphasis on the uniqueness of words to the document and low emphasis on the words with great appearance
   in the whole corpus of documents. 


#### 2. **Modelling**:

   RandomForestClassifier is used at this case, as it requires less preprocessing and have less model assumptions.
   One other benefit is that the Random Forest is robust to over-fitting, now matter of how many estimators used.
 

## Installing

To clone the git repository:

git clone https://github.com/MoustafaaAshraf/Disaster-Response-Message-Classification

## Files within App

- data
1. process_data.py: Python scripts for ETL (Extract, Transform and Load).
2. disaster_categories/disaster_messages: csv Data to be processed and used later to training a model.
   
- models
1. train_classifier.py: Python scripts for preprocessing and building a model.
2. classifier.pkl: Saved/trained model.

- app
1. master.html: a main page of web app.
2. go.html: classification result page of web app.
3. run.py: Python scripts for Flask file that runs app.

- README.md


## Executing Program

You can run the following commands in the project's directory to set up the database, train model and save the model.

1. To run ETL pipeline to clean data and store the processed data in the database
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

2. To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file 
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

3. Run the following command in the app's directory to run your web app. 
   python run.py

4. To view the App page to 
   http://0.0.0.0:3001/ OR http://localhost:3001/
   
## ScreenShots


