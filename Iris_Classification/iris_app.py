import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("🌸 Iris Flower Classification App")
st.write("Upload the Iris dataset to train the model and then classify custom flower measurements.")

uploaded_file = st.file_uploader("Upload IRIS.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📂 Dataset Preview")
    st.write(df.head())

    # Encode target column
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])

    X = df.drop('species', axis=1)
    y = df['species']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)

    st.subheader("📊 Model Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:\n" + classification_report(y_test, y_pred))

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Greens", ax=ax)
    st.pyplot(fig)

    # ----- User Input -----
    st.subheader("🌼 Predict Flower Species")

    sepal_length = st.slider("Sepal Length (cm)", float(df['sepal_length'].min()), float(df['sepal_length'].max()), float(df['sepal_length'].mean()))
    sepal_width = st.slider("Sepal Width (cm)", float(df['sepal_width'].min()), float(df['sepal_width'].max()), float(df['sepal_width'].mean()))
    petal_length = st.slider("Petal Length (cm)", float(df['petal_length'].min()), float(df['petal_length'].max()), float(df['petal_length'].mean()))
    petal_width = st.slider("Petal Width (cm)", float(df['petal_width'].min()), float(df['petal_width'].max()), float(df['petal_width'].mean()))

    user_data = pd.DataFrame({
        "sepal_length": [sepal_length],
        "sepal_width": [sepal_width],
        "petal_length": [petal_length],
        "petal_width": [petal_width]
    })

    if st.button("Predict Flower Species"):
        prediction = model.predict(user_data)[0]
        species_name = le.inverse_transform([prediction])[0]
        st.success(f"🌸 The flower species is **{species_name}**")
