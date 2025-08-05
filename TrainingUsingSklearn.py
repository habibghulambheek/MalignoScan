import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
reader = pd.read_csv("data.csv")
reader["diagnosis"] = reader["diagnosis"].map({"M":1,"B":0 })
y =  reader["diagnosis"]
x = reader.loc[:,"radius_mean":"fractal_dimension_worst"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
x_train = x_train.values
y_train  = y_train.values
x_test = x_test.values
y_test = y_test.values

model = LogisticRegression()
scalar = StandardScaler()
x_norm = scalar.fit_transform(x_train)
model.fit(x_norm,y_train)
x_test_norm = scalar.transform(x_test)
y_pred = model.predict(x_test_norm)
print('-' * 30)
print('Sklearn Logistic Regression')
print('-' * 30)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall (sensitivity):", recall_score(y_test, y_pred))
print("percision:", precision_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
confusion_mat = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print("Actual Value\tNo\tYes")
print(f"No\t\t{confusion_mat[0][0]}\t{confusion_mat[0][1]}")
print(f"Yes\t\t{confusion_mat[1][0]}\t{confusion_mat[1][1]}")
print('-' * 30)

