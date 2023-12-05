import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import sweetviz as sv
import os


def verify_file_html(file_name):
    _, extension = os.path.splitext(file_name)
    return extension.lower() == ".html"


def sweetviz_report(df):
    sw_report = sv.analyze(df)
    sw_report.show_html()


def rd_csv():
    df = pd.read_csv("./housing_price_dataset.csv")
    return df


def main():
    df = rd_csv()

    feature_col_names = ['SquareFeet', 'Bedrooms',
                         'Bathrooms', 'YearBuilt']
    df_encoded = pd.get_dummies(df, columns=['Neighborhood'], drop_first=True)
    target_col_name = 'Price'

    X = df_encoded[feature_col_names]
    y = df_encoded[target_col_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = MLPRegressor(activation='logistic', max_iter=1000,
                         hidden_layer_sizes=(4,), alpha=0.001, solver='lbfgs')
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f'R² Score: {score}')

    predictions = model.predict(X_test)

    mse = metrics.mean_squared_error(y_test, predictions)
    print("Erro quadrátrico médio:", mse)


if __name__ == "__main__":
    main()
