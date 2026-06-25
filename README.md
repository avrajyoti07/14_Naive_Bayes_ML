# 🚢 Titanic Survival Prediction — Naive Bayes Classifier

A machine learning project that predicts passenger survival on the Titanic using a **Gaussian Naive Bayes** classifier built with Python and scikit-learn.

---

## 📌 Project Overview

This project applies the Naive Bayes algorithm to the classic Titanic dataset to classify whether a passenger survived or not. It covers the full ML pipeline — from data cleaning and feature engineering to model training and prediction.

---

## 📂 Files

| File | Description |
|------|-------------|
| `13Naive_Bayes_Classifier.ipynb` | Main Jupyter Notebook with the complete ML pipeline |
| `Titanic-Dataset.csv` | Titanic passenger dataset (891 rows, 12 columns) |

---

## 📊 Dataset

The Titanic dataset contains information about **891 passengers** with the following features:

| Column | Description |
|--------|-------------|
| `PassengerId` | Unique passenger ID |
| `Survived` | Target variable — 0 = No, 1 = Yes |
| `Pclass` | Ticket class (1st, 2nd, 3rd) |
| `Name` | Passenger name |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | # of siblings/spouses aboard |
| `Parch` | # of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation |

**Overall survival rate: ~38.4%**

---

## 🔧 Steps Performed

### 1. Data Loading
Loaded the Titanic CSV dataset using `pandas`.

### 2. Feature Selection
Dropped irrelevant or high-cardinality columns that don't contribute meaningfully to prediction:
```
PassengerId, Name, SibSp, Parch, Ticket, Cabin, Embarked
```

### 3. Feature Engineering
- Converted the categorical `Sex` column into numeric dummy variables using `pd.get_dummies()`
- Removed the original `Sex` column after encoding

### 4. Handling Missing Values
- Identified missing values in the `Age` column
- Imputed missing ages with the **mean age** of all passengers

### 5. Train-Test Split
Split the data into training and testing sets using an **80/20 ratio**:
- Training samples: ~712
- Testing samples: ~179

### 6. Model Training
Used `GaussianNB` from `sklearn.naive_bayes` — suitable for continuous/numerical features that follow a Gaussian (normal) distribution.

### 7. Evaluation & Prediction
- Evaluated model accuracy using `model.score()`
- Made predictions with `model.predict()`
- Inspected probability outputs with `model.predict_proba()`

---

## 🧠 Key Concepts

**Why Naive Bayes?**  
Naive Bayes is a fast and simple probabilistic classifier based on Bayes' Theorem with an assumption of feature independence. Despite its "naive" assumption, it performs well on small-to-medium datasets and is highly interpretable.

**Why Gaussian NB?**  
Gaussian Naive Bayes assumes features follow a normal distribution — appropriate here since `Age`, `Fare`, and `Pclass` are continuous numerical values.

**Single `[ ]` vs Double `[[ ]]` brackets in pandas/sklearn:**  
- `df['col']` → returns a **1D Series** (fine for display, breaks sklearn)  
- `df[['col']]` → returns a **2D DataFrame** (required by sklearn's `fit()`, `transform()`)

---

## 🛠️ Tech Stack

- **Python 3**
- **pandas** — data manipulation
- **scikit-learn** — ML model (`GaussianNB`, `train_test_split`)
- **Jupyter Notebook** — interactive development

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas scikit-learn jupyter
```

### Run the Notebook
```bash
git clone https://github.com/your-username/titanic-naive-bayes.git
cd titanic-naive-bayes
jupyter notebook 13Naive_Bayes_Classifier.ipynb
```

> Make sure `Titanic-Dataset.csv` is in the same directory as the notebook.

---

## 📈 Results

The model is evaluated on 20% held-out test data. Run the notebook to see the accuracy score and per-sample predictions with probabilities.

---

## 🤝 Contributing

Pull requests are welcome! If you'd like to improve the model (e.g., add more features, try different classifiers, or tune hyperparameters), feel free to fork and submit a PR.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
