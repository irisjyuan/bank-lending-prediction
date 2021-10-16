import pandas as pd
import numpy as np
from os.path import exists
from tensorflow import keras
import tensorflow as tf
from model import MyModel


def create_train_df():
    mapping = {}

    # get from cache if possible
    if exists('train.pkl'):
        df = pd.read_pickle('train.pkl')
    else:
        df = pd.read_csv('lending_train.csv')
        df.to_pickle('train.pkl')

    # trim everything
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    # remove ID since it's not a factor
    df.pop('ID')

    # change loan_duration into numbers
    df['loan_duration'] = df['loan_duration'].str.replace(
        r'\D', '', regex=True)
    df['loan_duration'] = df['loan_duration'].astype(int)

    # categories eveything
    for header in list(df.columns.values):
        if df.dtypes[header] == 'object':
            df[header] = df[header].astype('category')
            mapping[header] = {k: v for v, k in enumerate(
                df[header].cat.categories)}
            df[header] = df[header].cat.codes

    return [df, mapping]


def create_predict_df():
    df = pd.read_csv('lending_topredict.csv')
    df.to_pickle('predict.pkl')
    return df


def create_submission(df):
    submission = pd.DataFrame(data=df['ID'])
    submission['loan_paid'] = np.random.randint(0, 2, submission.shape[0])
    submission.to_csv('submission.csv', index=False)


def create_model(df):
    # target column
    loan_paid = df.pop('loan_paid')

    # turn stuff into dictionaries
    train_dict = tf.data.Dataset.from_tensor_slices((dict(df), loan_paid))

    tf.convert_to_tensor(train_dict)
    model = MyModel()

    # model = keras.models.Sequential([keras.layers.Flatten(), keras.layers.Dense(
    #     768, activation=tf.nn.relu), keras.layers.Dense(10, activation=tf.nn.softmax)])
    # opt = keras.optimizers.Adam(learning_rate=0.0009)
    # model.compile(optimizer=opt,
    #               loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit(df, epochs=15)
    # return model


def run_model(df, model):
    return None


def main():
    train_df = None
    predict_df = None

    # pre-processing train df
    [train_df, mapping] = create_train_df()
    model = create_model(train_df)

    exit()

    if exists('predict.pkl'):
        predict_df = pd.read_pickle('predict.pkl')
    else:
        predict_df = create_predict_df()

    # create_submission(predict_df)


if __name__ == "__main__":
    main()
