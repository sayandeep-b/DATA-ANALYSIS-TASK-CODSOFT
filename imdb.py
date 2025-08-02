import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

file_path = r"C:\Users\SAYANDEEP\Desktop\CODSOFT DATASET!\IMDb Movies India.csv"
df = df = pd.read_csv(file_path, encoding='latin1')

print("First 5 rows of the dataset:\n", df.head())
print("\nColumns in dataset:\n", df.columns)

print("\nMissing values:\n", df.isnull().sum())

df = df.dropna(subset=['IMDB Rating'])

df['Genre'] = df['Genre'].fillna('Unknown')
df['Director'] = df['Director'].fillna('Unknown')
df['Star'] = df['Star'].fillna('Unknown')

label_enc = LabelEncoder()
df['Genre'] = label_enc.fit_transform(df['Genre'])
df['Director'] = label_enc.fit_transform(df['Director'])
df['Star'] = label_enc.fit_transform(df['Star'])

features = ['Genre', 'Director', 'Star']
target = 'IMDB Rating'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
