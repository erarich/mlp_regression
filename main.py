import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix  # Matriz de dispers�o
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
import sweetviz as sv
import os
import csv


def sweetviz_report(df, file_name):
    sw_report = sv.analyze(df)
    sw_report.show_html(filepath=file_name)


def rd_csv():
    df = pd.read_csv("./housing_price_dataset.csv")
    return df


def organize_dataset():
    df = rd_csv()

    feature_col_names = ['SquareFeet', 'Bedrooms',
                         'Bathrooms', 'YearBuilt']
    df_encoded = pd.get_dummies(df, columns=['Neighborhood'], drop_first=True)
    target_col_name = 'Price'

    X = df_encoded[feature_col_names]
    y = df_encoded[target_col_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_model(model, X_test, y_test):
    score = model.score(X_test, y_test)
    print(f'R² Score: {score}')

    predictions = model.predict(X_test)

    mse = metrics.mean_squared_error(y_test, predictions)
    mae = metrics.mean_absolute_error(y_test, predictions)
    print("Erro quadrátrico médio:", mse)
    print(f'O Erro Absoluto Médio (MAE) é: {mae}')

    return score, predictions, mse


def grid_search(X_train, y_train):
    param_grid = {
        'activation': ['logistic'],
        'max_iter': [1000, 2000, 3000, 4000, 5000],
        'hidden_layer_sizes': [(4,), (8,), (16,)],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'solver': ['lbfgs']
    }

    model = MLPRegressor()

    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Imprimir os melhores hiperparâmetros encontrados
    print("Melhores hiperparâmetros:")
    print(grid_search.best_params_)

    return grid_search.best_estimator_


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    mse = metrics.mean_squared_error(y_test, predictions)
    mae = metrics.mean_absolute_error(y_test, predictions)

    print(f'R² Score: {score}')
    print("Erro quadrátrico médio:", mse)
    print(f'O Erro Absoluto Médio (MAE) é: {mae}')

    with open('resultados.csv', 'a', newline='') as csvfile:
        fieldnames = ['Modelo', 'Activation', 'Max_iter',
                      'Hidden_Layer_Sizes', 'Alpha', 'Solver', 'Score', 'MSE', 'MAE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if os.stat('resultados.csv').st_size == 0:
            writer.writeheader()

        writer.writerow({
            'Modelo': model_name,
            'Activation': model.activation,
            'Max_iter': model.max_iter,
            'Hidden_Layer_Sizes': model.hidden_layer_sizes,
            'Alpha': model.alpha,
            'Solver': model.solver,
            'Score': score,
            'MSE': mse,
            'MAE': mae
        })

    df = pd.DataFrame({'y_test': y_test, 'predictions': predictions})
    df.insert(0, 'Index', range(1, len(df['y_test']) + 1))
    sweetviz_report(df, 'y_testPredictions.html')
    df.to_csv('output.csv', index=False)


def main():
    X_train, X_test, y_train, y_test = organize_dataset()

    model = MLPRegressor(activation='logistic', max_iter=5000,
                         hidden_layer_sizes=(8,), alpha=0.1, solver='lbfgs')

    train_and_evaluate_model(model, X_train, X_test,
                             y_train, y_test, 'NewModel')


if __name__ == "__main__":
    main()
