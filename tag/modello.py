import os
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, "csv", "Salary Data.csv")

df = pd.read_csv(csv_path)
df.dropna(axis=0, inplace=True)

x = df["Years of Experience"]
y = df["Salary"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

linear_regressor = LinearRegression()

#Addestramento e predizione
linear_regressor.fit(x_train.to_frame(), y_train)
y_test_pred = linear_regressor.predict(x_test.to_frame())

#Risultati
print(mean_absolute_percentage_error(y_test, y_test_pred) * 100)

# Predire i salari utilizzando il modello di regressione lineare
df['Predicted Salary'] = linear_regressor.predict(df[['Years of Experience']])

# Creare lo scatter plot con la retta di regressione
fig = px.scatter(df, x="Years of Experience", y="Salary", title="Scatter Plot con Retta di Regressione")
fig.add_scatter(x=df["Years of Experience"], y=df["Predicted Salary"], mode='lines', name='Retta di Regressione')
fig.show()