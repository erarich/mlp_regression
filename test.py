import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import sweetviz as sv

df = pd.read_csv('housing_price_dataset.csv')

# Separando os atributos de entrada (X) e o alvo (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Definindo os atributos categóricos e numéricos
categorical_features = ['Neighborhood']
numeric_features = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt']

# Criando um transformador para aplicar as transformações necessárias
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Criando o pipeline com o transformador e o modelo MLP
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MLPRegressor(random_state=42))
])

# Treinando o modelo
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Erro Quadrático Médio: {mse:.2f}')
print(f'R²: {r2:.2f}')

daf = pd.DataFrame({'y_test': y_test, 'predictions': y_pred})
daf.insert(0, 'Index', range(1, len(daf['y_test']) + 1))
daf.to_csv('output.csv', index=False)

sw_report = sv.analyze(daf)
sw_report.show_html(filepath="output.csv.html")
