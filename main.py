import pandas as pd
import numpy as np
from os.path import exists
from tensorflow import keras


def create_train_df():
    df = pd.read_csv('lending_train.csv')
    df.to_pickle('train.pkl')
    return df


def create_predict_df():
    df = pd.read_csv('lending_topredict.csv')
    df.to_pickle('predict.pkl')
    return df


def create_submission(df):
    submission = pd.DataFrame(data=df['ID'])
    submission['loan_paid'] = np.random.randint(0, 2, submission.shape[0])
    submission.to_csv('submission.csv', index=False)


def main():
    train_df = None
    predict_df = None

    # Reading from cache
    if exists('train.pkl'):
        train_df = pd.read_pickle('train.pkl')
    else:
        train_df = create_train_df()

    if exists('predict.pkl'):
        predict_df = pd.read_pickle('predict.pkl')
    else:
        predict_df = create_predict_df()

    create_submission(predict_df)


if __name__ == "__main__":
    main()
