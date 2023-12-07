from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import sweetviz as sv
import os
import csv
import seaborn as sns


def sweetviz_report(df, file_name):
    sw_report = sv.analyze(df)
    sw_report.show_html(filepath=file_name)


def rd_csv():
    df = pd.read_csv("./housing_price_dataset.csv")
    return df


def organize_dataset(data):
    colunas_numericas = ['SquareFeet', 'Bedrooms',
                         'Bathrooms', 'YearBuilt', 'Price']
    colunas_categoricas = ['Neighborhood']

    dados_numericos = data[colunas_numericas]
    dados_categoricos = data[colunas_categoricas]

    labelEncoder = LabelEncoder()

    dados_categoricos_transformados = dados_categoricos.apply(
        labelEncoder.fit_transform)
    dados_transformados = pd.concat(
        [dados_numericos, dados_categoricos_transformados], axis=1)

    onehotencorder = ColumnTransformer(
        transformers=[("OneHot", OneHotEncoder(), [5])], remainder='passthrough')

    dados_transformados = onehotencorder.fit_transform(
        dados_transformados).toarray()

    dados_transformados = np.delete(dados_transformados, 4, axis=1)
    classe = data['Price']

    train_x, test_valid_x, train_y, test_valid_y = train_test_split(
        dados_transformados, classe, test_size=0.1, random_state=0, stratify=dados_transformados[:, 0])

    test_x, valid_x, test_y, valid_y = train_test_split(
        test_valid_x, test_valid_y, test_size=0.5, random_state=0, stratify=test_valid_x[:, 0])

    model_first = tf.keras.models.Sequential()
    model_first.add(Dense(20, activation='relu',
                    kernel_initializer='he_uniform'))
    model_first.add(Dense(1, activation='relu'))
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model_first.compile(loss='mape', optimizer=opt, metrics=['mae'])
    history_tiny = model_first.fit(train_x, train_y,
                                   validation_data=(valid_x, valid_y),
                                   epochs=200, verbose=1, batch_size=train_x.shape[0])
    title_tiny = 'Métrica do modelo Tiny'
    plotHistory(history_tiny, title_tiny)


def plotHistory(history, title):
    fig, ax = plt.subplots(1, 2, figsize=(26, 10))

    # Imprime a curva de aprendizado
    ax[0].set_title('Mean Absolute Percentage Error', pad=-40)
    ax[0].plot(history.history['loss'], label='train')
    ax[0].plot(history.history['val_loss'], label='valid')
    ax[0].legend(loc='best')

    # Imprime a curva de acurácia
    ax[1].set_title('Mean Absolute Error', pad=-40)
    ax[1].plot(history.history['mae'], label='train')
    ax[1].plot(history.history['val_mae'], label='valid')
    ax[1].legend(loc='best')

    fig.suptitle(title)

    fig.savefig(
        '' + title)
    plt.show()


def main():
    data = rd_csv()
    organize_dataset(data)


if __name__ == "__main__":
    main()
