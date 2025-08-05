import pandas as pd
from LogisticRegression import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix,recall_score
from matplotlib import  pyplot
reader = pd.read_csv("data.csv")
reader["diagnosis"] = reader["diagnosis"].map({"M":1,"B":0 })
y =  reader["diagnosis"]

x = reader.loc[:,"radius_mean":"fractal_dimension_worst"]

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)
x_train = x_train.values
y_train  = y_train.values
x_test = x_test.values
y_test = y_test.values

x_norm,_x_mean, _x_std = z_score(x_train)
x_test_norm = scale_input(x_test,_x_mean,_x_std)
n  = x_norm.shape[1] # number of features
w_in =  np.zeros(n)
b_in  = 0.0

w_out,b_out, j_hist, w_hist, b_hist =  gradient_descent(x_norm,y_train, w_in,b_in,0.1,0, 1000)

y_pred = f_wb(z(x_test_norm,w_out,b_out))
y_pred = (y_pred >= 0.5)
y_pred.astype('int')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall (sensitivity):", recall_score(y_test, y_pred))
print("percision:", precision_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
confusion_mat = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print("Actual Value\tNo\tYes")
print(f"No\t\t{confusion_mat[0][0]}\t{confusion_mat[0][1]}")
print(f"Yes\t\t{confusion_mat[1][0]}\t{confusion_mat[1][1]}")

np.savez("Cancer_predictor_parameters.npz",w = w_out, b = b_out, x_mean =_x_mean,x_std=_x_std)
pyplot.plot(j_hist)
pyplot.xlabel("No of iterations")
pyplot.ylabel("Cost")
pyplot.title("Cost of model w.r.t iterations")
pyplot.show()