# 🎗 Cancer Detection Using Logistic Regression

This is a Machine Learning project that detects breast cancer using the Wisconsin dataset.  
It includes two models:  
- A **hand-coded Logistic Regression** from scratch  
- A model using **Scikit-learn's LogisticRegression**

The aim is to understand the working of Logistic Regression in depth and evaluate how closely a self-implemented version performs against a library model.

---

## 💡 Project Motivation

After studying Logistic Regression, overfitting, underfitting, generalization, and evaluation metrics like F1 Score, Precision, and Recall, I wanted to build a practical ML project without relying entirely on built-in tools.

This helped deepen my understanding of how Logistic Regression functions internally.

---

## ✅ Problem Solved

Breast cancer is one of the most common types of cancer in women.  
Early diagnosis can **significantly increase survival rates**.  
This project uses **ML classification** to predict the chances of a tumor being maliginant(cancerous).

---

## 📚 Learning Outcomes

While building this project, I:

- ✅ Implemented Logistic Regression manually using gradient descent
- ✅ Understood the logic behind weight updates
- ✅ Applied regularization and feature scaling, and learned why they're important
- ✅ Used scikit-learn to compare results and select high-impact features (those contributing more than 90% to predictions out of 30) 
- ✅ Learned how to evaluate models with:
  - Confusion Matrix  
  - Accuracy  
  - F1 Score  
  - Precision  
  - Recall  

- ✅ Gained deeper understanding of how even small changes in data can affect the outcome  
- ✅ Applied StandardScaler to normalize features

---

## 📊 Evaluation Results

| Metric              | Custom Model | Scikit-learn Model |
|---------------------|--------------|--------------------|
| Accuracy            | 98.24%       | 97.36%             |
| Recall (Sensitivity)| 97.67%       | 95.35%             |
| Precision           | 97.67%       | 97.61%             |
| F1 Score            | 97.67%       | 96.47%             |

### 🧾 Confusion Matrices

#### Custom Logistic Regression

| Actual \ Predicted | No  | Yes |
|---------------------|-----|-----|
| No                  | 70  | 1   |
| Yes                 | 1   | 42  |

#### Scikit-learn LogisticRegression

| Actual \ Predicted | No  | Yes |
|---------------------|-----|-----|
| No                  | 70  | 1   |
| Yes                 | 2   | 41  |

Run TrainingUsingLogisticRegression.py and TrainingUsingSklearn.py to generate and compare results.
---

## ⚙️ Installation Instructions

1. Clone the repository:
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:
   ```bash
   python main.py
   ```

---

## 🧠 Usage Instructions

* Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
* CSV file: `data.csv` (included in repo)
* Steps:

  * Run `main.py`

  * You can manually input values or load from a `.txt` file containing comma-separated values:

    ```
    perimeter_worst, concave points_worst, concave points_mean, area_worst, radius_worst, area_mean,
    concavity_mean, area_se, radius_mean, perimeter_mean, concavity_worst, perimeter_se, texture_worst,
    radius_se, compactness_worst
    ```

    Example:

    ```
    152.5,0.2430,0.12790,209.0,23.57,303.0,0.1974,34.03,19.69,123.4,0.6869,4.585,25.53,0.7456,0.8663
    ```

  * The model predicts the **percentage chance of malignancy**.

---

## 📂 Folder Structure

MalignoScan/
│
├── README.md                         # Project documentation  
├── requirement.txt                   # Required Python packages  
├── data/                             # Main dataset folder  
├── RecordsForTesting/                # Folder containing individual test record files  
│   ├── record1.txt ... record10.txt  
├── main.py                           # Main entry point to run the project  
├── TrainingUsingSklearn.py           # ML model training using scikit-learn  
├── TrainingUsingLogisticRegression.py# ML model training using custom logistic regression  
├── LogisticRegression.py             # Custom Logistic Regression implementation  
├── Finding_Important_features.py     # Feature selection logic  
├── PatternsInDataCsv.py              # Code to explore patterns in the dataset  
├── Imp_ftrs_parameters.npz           # Saved model parameters (important features)  
├── Cancer_predictor_parameters.npz   # Saved model parameters (trained weights)  
├── data.ico                          # Application icon  


## 👤 Credits

This project was fully built by me, **Habib**, as a step toward mastering Machine Learning through real understanding — not shortcuts.  
No copy-paste. Just math, code, and lots of testing.

---

## 🤝 Contributions Welcome

Feel free to:
- Suggest optimizations for the scratch model  
- Report issues  
- Try it on other datasets!
