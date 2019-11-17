import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.externals import joblib

def load_data(database_filepath):
    '''
    Input:
          database_filepath : input sqllite database filepath

    Output:
           X : meassage data
           Y : categories data
           category_names : label for 36 columns 
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names

def tokenize(text):
    
    '''
    Input:
          text : input text that to be processed

    Output:
          clean_words : array of clean words that was tokenized, lowercase, stripped 
    '''

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_words = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_words.append(clean_tok)

    return clean_words


def build_model():
    
    '''
    Build a Machine Learning pipeline using tfidf, gridcv and random forest

    Input :
            None

    Output :
            gridsearch_cv : results of gridsearch cv
    '''
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    parameters = {'clf__estimator__min_samples_split': [2,4],
                 }
    gridsearch_cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return gridsearch_cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Evaluate a Trained model on test dataset

    Input :
           model : trained model
           X_test : input testing set
           Y_test : label testing set
           category_names : label for 36 columns

    Output :
            return model performance stats
    '''

    preds = model.predict(X_test)
    for label in range(0, len(category_names)):
        print('Message category:', category_names[label])
        print(classification_report(Y_test[:, label], preds[:, label]))



def save_model(model, model_filepath):
    '''
        Input:
              model : trained model

        Output:
              model_filepath : save model as pickle file

    '''

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