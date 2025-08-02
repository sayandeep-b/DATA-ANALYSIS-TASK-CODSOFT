import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

file_path = r"C:\Users\SAYANDEEP\Desktop\TITANIC DATASET!\Titanic-Dataset.csv"
df = pd.read_csv(file_path)

print(df.head())

print("\nMissing values:\n", df.isnull().sum())

df = df.drop(['Cabin', 'Name', 'Ticket'], axis=1)

df['Age'].fillna(df['Age'].median(), inplace=True)

df.dropna(inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sns.barplot(x="Sex", y="Survived", data=df)
plt.title("Survival Rate by Sex")
plt.xticks([0, 1], ['Male', 'Female'])
plt.show()
