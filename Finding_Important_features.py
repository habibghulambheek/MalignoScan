import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

x_cols =  np.array(["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"])

df = pd.read_csv("data.csv")
y = df["diagnosis"]
x = df.drop(["diagnosis","id"], axis  = 1)
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)
model = RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)
importance  = model.feature_importances_
imp_cols = list(zip(x_train.columns, importance))
imp_cols.sort(key = lambda x:x[1],reverse=True)
imp = 0
# predicting important columns and how much each contribute to prediction, 
# we only take the first parameters that contribute more than 90 percent to the prediction 
threshold = 0.90
required_parameters =  0
col_names =  []
for i in range(len(imp_cols)):
    print(f"{imp_cols[i][0]}: {imp_cols[i][1]}")
    col_names.append(imp_cols[i][0])
    imp += imp_cols[i][1]
    required_parameters += 1
    if imp > threshold:
        break


print("->Saving the data in a npz file.")

parameters  = np.load("Cancer_predictor_parameters.npz")
w =  parameters["w"]
b_in = parameters["b"]
_x_mean = parameters["x_mean"]
_x_std = parameters["x_std"]

xw = dict(zip(x_cols, w))
xx_mean = dict(zip(x_cols, _x_mean))
xx_std = dict(zip(x_cols, _x_std))
w_new =  np.zeros(required_parameters)
x_mean_new =  np.zeros(required_parameters)
x_std_new =  np.zeros(required_parameters)

for i in range(required_parameters):
    w_new[i]  =  xw[imp_cols[i][0]]
    x_mean_new[i]  =  xx_mean[imp_cols[i][0]]
    x_std_new[i]  =  xx_std[imp_cols[i][0]]
    print(imp_cols[i][0],w_new[i])
    print(imp_cols[i][0],x_mean_new[i])
    print(imp_cols[i][0],x_std_new[i])
np.savez("Imp_ftrs_parameters.npz",columns =  col_names,w = w_new, b = b_in, x_mean = x_mean_new, x_std = x_std_new)