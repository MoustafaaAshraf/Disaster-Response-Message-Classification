# importing needed libraries
import sys
import pandas as pd
import re
from sqlalchemy import create_engine
import pickle

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    loading the dataframe from the database file
    :param database_filepath: location of the saved dataframe
    :return:
    X: Dataframe of the messages used for predictions
    y: Dataframe of the multi-label used to train the model
    categories_name: a list of the names of the multiclass targets used to train the model
    '''

    # establishing the connection with the database
    engine = create_engine('sqlite:///' + database_filepath)

    # reading the model from the SQL database
    df = pd.read_sql('etltable', engine)

    # Slicing the dataframe to acquire the targets and labels
    X = df.loc[:, 'message']
    y = df.iloc[:, 4:]

    # Extracting the names of the target columns
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    '''
    turing plain text into cleaned list of tokens for cleaned words
    :param text: a plain text message
    :return: clean_tokens: a list of cleaned tokens from the text, cleaned from punctuations, stopwords and words are
    lemmatized
    '''

    # Removing the punctuation from text by keeping only a->z, A->Z and 0-9 characters from the text
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # Splitting the text into tokens/words through the word_tokenize method from nltk library
    tokens = word_tokenize(text)

    # list comprehension checking every word in the list and choosing the words which are not stop words
    tokens = [w for w in tokens if w not in stopwords.words('english')]

    # Instanciating an empty list for cleaned tokens
    clean_tokens = []

    # looping through tokens
    for tok in tokens:
        # getting lower case and removing the spaces from words
        clean_tok = tok.lower().strip()
        # adding the tokens to the cleaned_tokens list
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    creating a pipeline for modelling the database of categories and labels
    :return: model to be trained using grid search
    '''
    pipeline = Pipeline([
        # creating a count vectorizer using the tokens provided by a list
        ('vect', CountVectorizer(tokenizer=tokenize)),

        # Providing the tfidf transformer from sklearn
        ('tfidf', TfidfTransformer()),

        # an instance of multi-class classifier based on RandomForestClassifier
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=50)))
    ])

    parameters = {'clf__estimator__max_depth': [10, None],
                  'clf__estimator__min_samples_leaf': [5, 10]}

    grid = GridSearchCV(pipeline, parameters)

    return grid


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    a method used for evaluating the performance of the model by printing the metrics for classification
    :param model: pipeline for preprocessing and modelling
    :param X_test: a dataframe of the labels used for predicting the targets
    :param Y_test: a dataframe of targets used to train the model
    :param category_names: a list of the names of targets used for training the model
    '''

    # Predicting the classification prediction of the created model
    preds = model.predict(X_test)

    # Iterating through the columns of the testing columns and comparing them to the predicted classification
    for idx, col in enumerate(category_names):

        # Printing the classification metrics
        print(col, classification_report(Y_test.iloc[:, idx], preds[:, idx]))


def save_model(model, model_filepath):
    '''
    Saving a file of the trained model
    :param model: trained model
    :param model_filepath: location of the trained model file
    '''

    # Creating a pathfile to save the model
    model_pkl = open(model_filepath, 'wb')

    # Saving the model into the created location
    pickle.dump(model, model_pkl)

    # Closing
    model_pkl.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()