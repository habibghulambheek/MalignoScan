from matplotlib import pyplot
import pandas as pd
import numpy as np
reader = pd.read_csv("data.csv")
reader["diagnosis"] = reader["diagnosis"].map({"M":1,"B":0 })
y_train =  reader["diagnosis"].values
x_train = reader.loc[:,"radius_mean":"fractal_dimension_worst"].values
x_cols =  np.array(["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"])

n = x_cols.shape[0]
cols = 6
rows = n//cols

fig,graphs = pyplot.subplots(rows,cols,figsize=(20,8), constrained_layout = True,sharey = True)
graphs = graphs.flatten()
for i in range(n):
    graphs[i].scatter(x_train[:, i], y_train)
    graphs[i].set_xlabel(x_cols[i])

graphs[0].set_ylabel("Diagnosis")

pyplot.show()