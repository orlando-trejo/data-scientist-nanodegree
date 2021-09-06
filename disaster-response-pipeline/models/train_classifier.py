import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import re

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    """
    Load data from SQL db file.

    Parameters
    ----------
    database_filepath : str
        Path to db file.

    Returns
    -------
    X : dataframe
        Features (messages) dataframe.
    Y : dataframe
        Labels (categories) dataframe.
    category_names : list
        Names of the different categories.

    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_response', engine)
    df.dropna(inplace=True)
    X = df.message
    Y = df.iloc[:,4:]
    Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)
    category_names = Y.columns
    print(Y.head())
    return X, Y, category_names


def tokenize(text):
    """
    Process text data by removing punctuation, lowering case, tokenizing words,
    remvoing stop words, and lemmetizing.

    Parameters
    ----------
    text : str
        Text data of messages.

    Returns
    -------
    tokens : list
        List of processed word tokens for a given message.

    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Build the pipeline and gridsearch for a machine learning model.

    Returns
    -------
    cv : GridSearchCV
        A model with a processing, classifier, and grid search piepline.

    """
    # build a machine learning pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    # use grid search to find better parameters
    parameters = {
    'clf__estimator__n_neighbors': [5, 10],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Create a classification report on a trained models for the test set.

    Parameters
    ----------
    model : trained model
        Trained model on processed text and categories.
    X_test : dataframe
        Test dataframe for features (messages).
    Y_test : dataframe
        Test dataframe for labels (categories).
    category_names : list
        List of the different categories.

    Returns
    -------
    None.

    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Export trained and tuned model to a pkl file.

    Parameters
    ----------
    model : Trained model
        Trained model on processed text and categories.
    model_filepath : str
        File path to export pkl file.

    Returns
    -------
    None.

    """
    pickle.dump(model, open(model_filepath, 'wb'))


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
