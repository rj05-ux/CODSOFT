# CODSOFT

This repository contains projects completed during my **Data Science Internship at CodSoft**. Each project is an interactive **Streamlit web app** that trains a machine learning model on a classic dataset and lets users make live predictions from custom input.

---

## 1. Iris Flower Classification
**Folder:** `Iris_Classification/`
**App:** `iris_app.py`
**Dataset:** `IRIS.csv`

### Overview
Classifies iris flowers into species based on sepal and petal measurements using a K-Nearest Neighbors (KNN) classifier.

### Features
- Upload the Iris dataset directly in the app
- Encodes the target `species` column with `LabelEncoder`
- Trains a `KNeighborsClassifier` (k = 5) on an 80/20 train-test split
- Displays accuracy, classification report, and a confusion matrix heatmap
- Interactive sliders for sepal/petal length & width to predict the species of a custom flower

### Tech Stack
`streamlit`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`

---

## 2. Sales Prediction with Advertising Data
**Folder:** `Sales_Prediction/`
**App:** `sales_app.py`
**Dataset:** `advertising.csv`

### Overview
Predicts product sales based on advertising spend across TV, Radio, and Newspaper channels using Linear Regression.

### Features
- Upload the advertising dataset directly in the app
- Trains a `LinearRegression` model on TV, Radio, and Newspaper spend vs. Sales
- Displays R² score and Mean Squared Error (MSE)
- Plots actual vs. predicted sales
- Interactive sliders for ad budgets to predict sales for a custom spend scenario

### Tech Stack
`streamlit`, `pandas`, `matplotlib`, `scikit-learn`

---

## 3. Titanic Survival Prediction
**Folder:** `Titanic_Survival/`
**App:** `titanic_app.py`
**Dataset:** `Titanic-Dataset.csv`

### Overview
Predicts whether a passenger would have survived the Titanic disaster using Logistic Regression.

### Features
- Upload the Titanic dataset directly in the app
- Drops non-predictive columns (`Name`, `Cabin`, `Ticket`)
- Handles missing values (median `Age`, mode `Embarked`)
- Encodes `Sex` and `Embarked` with `LabelEncoder`
- Trains a `LogisticRegression` model (excluding `PassengerId`) on an 80/20 split
- Displays accuracy and classification report
- Interactive inputs (class, sex, age, siblings/spouses, parents/children, fare, embarkation port) to predict survival for a custom passenger

### Tech Stack
`streamlit`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`

---

## Repository Structure
```
CODSOFT/
├── Iris_Classification/
│   └── iris_app.py
├── Sales_Prediction/
│   └── sales_app.py
├── Titanic_Survival/
│   └── titanic_app.py
├── IRIS.csv
├── advertising.csv
├── Titanic-Dataset.csv
└── README.md
```

## How to Run
1. Clone this repository
   ```bash
   git clone https://github.com/rj05-ux/CODSOFT.git
   cd CODSOFT
   ```
2. Install the required libraries
   ```bash
   pip install streamlit pandas seaborn matplotlib scikit-learn
   ```
3. Run any app with Streamlit, then upload the corresponding CSV when prompted in the browser:
   ```bash
   streamlit run Iris_Classification/iris_app.py
   streamlit run Sales_Prediction/sales_app.py
   streamlit run Titanic_Survival/titanic_app.py
   ```

## About
These projects were completed as part of the **CodSoft Data Science Internship**, focused on building end-to-end, interactive ML applications — from data preprocessing and model training to deployment as usable Streamlit apps.

## Author
**Rutuja Jadhav**
GitHub: [rj05-ux](https://github.com/rj05-ux)
LinkedIn: [rutuja-jadhav](https://linkedin.com/in/rutuja-jadhav-592388321)
