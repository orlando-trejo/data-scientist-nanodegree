import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load the messages and categories data.

    Parameters
    ----------
    messages_filepath : str
        Path to messages data file.
    categories_filepath : str
        Pato to categories data file.

    Returns
    -------
    dataframe
        Merged dataframe with messages and categories data.

    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """
    Clean the data by renaming the columns and categorizing into numerical
    values.

    Parameters
    ----------
    df : dataframe
        Merged dataframe without processed category columns.

    Returns
    -------
    df : dataframe
        Merged dataframe with renamed and numerical values for category columns.

    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[0:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert category values to just numbers 0 or 1
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = [int(val[-1]) for val in categories[column]]

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    Save data to SQL db file

    Parameters
    ----------
    df : dataframe
        Datframe to export to db file.
    database_filename : str
        Name of db file including '.db'.

    Returns
    -------
    None.

    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_response', engine, index=False)


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
