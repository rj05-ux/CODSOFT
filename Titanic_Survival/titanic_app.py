import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("🚢 Titanic Survival Prediction App")
st.write("Upload Titanic dataset to train the model and then test survival prediction for custom passengers.")

# Upload dataset
uploaded_file = st.file_uploader("Upload Titanic-Dataset.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Drop useless columns
    df = df.drop(columns=['Name', 'Cabin', 'Ticket'], errors='ignore')

    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Encode categorical features
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    # ✅ Drop PassengerId from training
    X = df.drop(['Survived', 'PassengerId'], axis=1, errors='ignore')
    y = df['Survived']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # ----- Show evaluation -----
    st.subheader("📊 Model Evaluation")
    y_pred = model.predict(X_test)
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:\n" + classification_report(y_test, y_pred))

    

    # ----- User Input -----
    st.subheader("🧍 Passenger Survival Prediction")

    pclass = st.selectbox("Ticket Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 25)
    sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
    fare = st.slider("Fare", 0.0, 500.0, 32.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

    # Encode user input
    sex_val = 1 if sex == "male" else 0
    embarked_val = {"S": 0, "C": 1, "Q": 2}[embarked]

    user_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex_val],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked_val]
    })

    if st.button("Predict Survival"):
        prediction = model.predict(user_data)[0]
        if prediction == 1:
            st.success("✅ The passenger would have SURVIVED!")
        else:
            st.error("❌ The passenger would NOT have survived.")
