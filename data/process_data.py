import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
        'message_filepath' : a string/path to a message csv file
        'categories_filepath' : a string/path to a categories csv file
        
    OUTPUT:   
        df : a merge dataset by id
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    '''
    INPUT 
        df: Dataframe to be cleaned by the method
    OUTPUT
        df: Returns a cleaned dataframe with new categories and drop duplicated:
    '''
    categories = df['categories'].str.split(';', expand=True)
    row = categories[:1]
    category_colnames = row.apply(lambda x: x[0][:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(-1)

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()

    return df
    

def save_data(df, database_filename):
    '''
    INPUT:
        - Cleaned data frame
        - database name to save
    OUTPUT:
        - None
    '''
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('DisasterResponse', engine, index=False)


def main():
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