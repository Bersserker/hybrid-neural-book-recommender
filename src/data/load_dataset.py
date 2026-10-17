import pandas as pd
from pathlib import Path
from src.config import  RAW_DATA_DIR


def laod_dataframes(path: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    df_books = pd.read_csv(path / 'books.csv')
    df_ratings = pd.read_csv(path / 'ratings.csv')
    df_book_tags = pd.read_csv(path / 'book_tags.csv')
    df_tags = pd.read_csv(path / 'tags.csv')
    df_to_read = pd.read_csv(path / 'to_read.csv')


    dataset_dict = {'df_books':df_books,
                'df_ratings':df_ratings,
                'df_book_tags':df_book_tags,
                'df_tags':df_tags,
                'df_to_read':df_to_read}
    return dataset_dict


if __name__ == "__main__":
    laod_dataframes()