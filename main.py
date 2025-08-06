import numpy as np
from tkinter import *
from tkinter import filedialog
import os
from LogisticRegression import f_wb, z,scale_input

curr_dir = os.getcwd()
# print(os.getcwd())
parameters = np.load(f"{curr_dir}\\Imp_ftrs_parameters.npz")
w = parameters["w"]
b = parameters["b"]
x_mean = parameters["x_mean"]
x_std = parameters["x_std"]


root = Tk()
root.title("MalignoScan🎗️")
root.iconbitmap("breast-cancer-awareness.ico")
root.geometry("1300x510")
root.configure(bg = "#f2f2f2")
root.resizable(False,False)

Label(root, text = "🎗 Welcome to MalignoScan 🎗", font = ("Arial",18,"bold"), pady = 0,fg="#880E4F",bg = "white", relief = 'ridge', bd = 3).pack(pady = 20)
mainFrame = LabelFrame(root,padx = 30, pady = 30, bg = "white")
mainFrame.pack()

# Giving more understandable names to all features
col_names = np.array([
    "Concave Points (Worst Case)",        # concave points_worst: 0.1572
    "Tumor Area (Worst Case)",            # area_worst: 0.1254
    "Concave Points (Average)",           # concave points_mean: 0.1214
    "Perimeter (Worst Case)",             # perimeter_worst: 0.1181
    "Radius (Worst Case)",                # radius_worst: 0.0798
    "Tumor Area (Average)",               # area_mean: 0.0584
    "Concavity (Average Depth)",          # concavity_mean: 0.0475
    "Radius (Average)",                   # radius_mean: 0.0344
    "Tumor Area Variation",               # area_se: 0.0342
    "Perimeter (Average)",                # perimeter_mean: 0.0283
    "Concavity (Worst Case)",             # concavity_worst: 0.0255
    "Texture Irregularity (Worst Case)",  # texture_worst: 0.0184
    "Perimeter Variation",                # perimeter_se: 0.0182
    "Texture Irregularity (Average)",     # texture_mean: 0.0172
    "Compactness (Worst Case)"            # compactness_worst: 0.0152
])


n = col_names.shape[0]

col_labels = []
col_entries = []
_row = 0
_col = 0
for i in range(n):
    
    col_labels.append(Label(mainFrame,text = f"Enter {col_names[i]}:", font = ('Arial',10),bg = "white", padx = 10, pady = 4))
    col_labels[i].grid(row = _row,column = _col,sticky = "w")
    col_entries.append(Entry(mainFrame, width = 25, relief="sunken", bd =  1.5))
    col_entries[i].grid(row = _row, column = _col+1)
    
    # print(_row,_col,"  " , _row, _col+1)
    _row += 1
    if _row == 7:
        _col += 2
    _row %= 7

def predict_malignancy():

    try:
        x = [float(col_entries[i].get()) for i in range(n)]
        x = scale_input(np.array(x),x_mean, x_std)
        # print(x)
    except  ValueError:
         result.config(text = "Please enter a valid numerical value.", fg =  'orange') 
         return  
    # print(x)
    z_val = z(x,w,b)
    y = f_wb(z_val)
    # print(z_val)
    # print(y)
    
    if y >= 0.5:
        result.config(text = f"The tumour has {(y*100):.2f}% chances of being malignant.", fg =  'red')
    else:
        result.config(text = f"The tumour has {(y*100):.2f}% chances of being malignant.", fg =  'green')


def browse_data():
    
    file_loc =  filedialog.askopenfilename(initialdir=curr_dir,title="Select your record", filetypes=(("Text files","*.txt"),))
    try:
        with open(file_loc, 'r') as text_file: # opening the file
            record = text_file.read() # reading the content
    
        record = record.split(',')
        m  = len(record)
        if m != n:
             raise ValueError(f"Expected {n} values, but found {m} in the file.")
        for i in range(n):
            col_entries[i].delete(0,'end')
            col_entries[i].insert(0, record[i])
    except Exception as e:
        result.config(text = f"Please select a valid record file.\n Error: {e}", fg =  'orange')



predict = Button(root, text="Predict Malignancy",bg="#880E4F", fg = "white", font= ("Segou UI", 15 , "bold"), padx = 5, pady = 10, command=predict_malignancy)
predict.pack(pady= 10)    

browse_btn =  Button(root, text = "Browse data", font = ("Arial", 10), command= browse_data)
browse_btn.pack()

result = Label(root, text = "", fg =  "red", font= ("Arial" ,  15, "bold"), padx = 5, pady = 5)
result.pack(pady  = 10)
root.mainloop()


# Maximum and minimum values found in the training set for each input feature
# radius_mean: Max= 28.11, Min= 6.981
# texture_mean: Max= 39.28, Min= 9.71
# perimeter_mean: Max= 188.5, Min= 43.79
# area_mean: Max= 2501.0, Min= 143.5
# smoothness_mean: Max= 0.1634, Min= 0.05263
# compactness_mean: Max= 0.3454, Min= 0.01938
# concavity_mean: Max= 0.4268, Min= 0.0
# concave points_mean: Max= 0.2012, Min= 0.0
# symmetry_mean: Max= 0.304, Min= 0.106
# fractal_dimension_mean: Max= 0.09744, Min= 0.04996
# radius_se: Max= 2.873, Min= 0.1115
# texture_se: Max= 4.885, Min= 0.3602
# perimeter_se: Max= 21.98, Min= 0.757
# area_se: Max= 542.2, Min= 6.802
# smoothness_se: Max= 0.03113, Min= 0.001713
# compactness_se: Max= 0.1354, Min= 0.002252
# concavity_se: Max= 0.396, Min= 0.0
# concave points_se: Max= 0.05279, Min= 0.0
# symmetry_se: Max= 0.07895, Min= 0.007882
# fractal_dimension_se: Max= 0.02984, Min= 0.0008948
# radius_worst: Max= 36.04, Min= 7.93
# texture_worst: Max= 49.54, Min= 12.02
# perimeter_worst: Max= 251.2, Min= 50.41
# area_worst: Max= 4254.0, Min= 185.2
# smoothness_worst: Max= 0.2226, Min= 0.07117
# compactness_worst: Max= 1.058, Min= 0.02729
# concavity_worst: Max= 1.252, Min= 0.0
# concave points_worst: Max= 0.291, Min= 0.0
# symmetry_worst: Max= 0.6638, Min= 0.1565
# fractal_dimension_worst: Max= 0.2075, Min= 0.05504
