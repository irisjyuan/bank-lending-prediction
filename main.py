import pandas as pd
import numpy as np
from os.path import exists
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import mod
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

    df = df.fillna(0)

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

    df = df.astype(int)

    return [df, mapping]


def create_predict_df(mapping):
    # get from cache if possible
    if exists('predict.pkl'):
        df = pd.read_pickle('predict.pkl')
    else:
        df = pd.read_csv('lending_topredict.csv')
        df.to_pickle('predict.pkl')

    # trim everything
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    df = df.fillna(0)

    # remove ID since it's not a factor
    df.pop('ID')
    df.pop('loan_paid')

    # change loan_duration into numbers
    df['loan_duration'] = df['loan_duration'].str.replace(
        r'\D', '', regex=True)
    df['loan_duration'] = df['loan_duration'].astype(int)

    # categories eveything
    for header in list(df.columns.values):
        if df.dtypes[header] == 'object':
            df[header] = df[header].map(mapping[header])

    df = df.fillna(0)
    df = df.astype(int)

    return df


def create_submission(pred):
    df = pd.read_csv('lending_topredict.csv')
    arr = pred.tolist()
    arr = [1 if s > 0.5 else 0 for [s] in arr]
    id_arr = df['ID'].tolist()

    data = {'ID': id_arr, 'loan_paid': arr}
    print(len(id_arr), len(arr))
    submission = pd.DataFrame(data=data)
    submission.to_csv('submission.csv', index=False)


def create_model(df):
    new_df = df.drop('loan_paid', axis=1)
    tensor = tf.convert_to_tensor(new_df)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(tensor)

    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(16, activation='sigmoid'),
        tf.keras.layers.Dense(4, activation='sigmoid'),
        tf.keras.layers.Dense(2, activation='sigmoid'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def train_model(df, model):
    target = df.pop('loan_paid')
    tensor = tf.convert_to_tensor(df)
    model.fit(tensor, target, epochs=50, batch_size=10000)
    return model


def run_model(df, model):
    pred = model.predict(df)
    return pred


def main():

    # # pre-processing train df
    [train_df, mapping] = create_train_df()

    model = create_model(train_df)
    model = train_model(train_df, model)
    print(model.summary())

    predict_df = create_predict_df(mapping)

    print(train_df.shape, predict_df.shape)

    pred = run_model(predict_df, model)
    print(pred)

    create_submission(pred)


if __name__ == "__main__":
    main()
