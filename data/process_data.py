import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Input:
          message_filepath : input csv file path of message data
          categories_data  : input csv file path of categories data
    Output:
          df : dataframe which contains merged data from both messages and categories

    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    '''

    Input:
          df : dataframe which contains merged data from both messages and categories

    Output:
          df : cleaned dataset of merged data
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories[:1]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.str.split('-')[0][0], axis=0)
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1] if int(x.split('-')[1]) < 2 else 1)
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)

    # check number of duplicates
    print(df.duplicated().sum())

    # drop duplicates
    df = df.drop_duplicates(keep='first')

    # check number of duplicates
    print(df.duplicated().sum())

    return df


def save_data(df, database_filename):
    '''

    Input:
          df : cleaned dataset of merged data

    Output:
          save cleaned dataset of merged data to sqllite database
    '''

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    print(sys.argv)
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()