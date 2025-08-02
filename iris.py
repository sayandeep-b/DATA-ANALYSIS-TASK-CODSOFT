import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

file_path = r"C:\Users\SAYANDEEP\Desktop\CODSOFT DATASET!\IRIS.csv"
df = pd.read_csv(file_path)

print("Dataset preview:\n", df.head())

print("\nMissing values:\n", df.isnull().sum())

label_enc = LabelEncoder()
df['species'] = label_enc.fit_transform(df['species']) 

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sns.pairplot(df, hue='species', diag_kind='kde')
plt.suptitle("Iris Flower Feature Distribution", y=1.02)
plt.show()
