import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("📈 Sales Prediction with Advertising Data")
st.write("Upload the advertising dataset to train the model and predict future sales.")

# File uploader
uploaded_file = st.file_uploader("Upload advertising.csv", type=["csv"])

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("📂 Dataset Preview")
    st.write(df.head())

    # Features and target
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # ----- Evaluation -----
    st.subheader("📊 Model Evaluation")
    st.write("R² Score:", r2_score(y_test, y_pred))
    st.write("MSE:", mean_squared_error(y_test, y_pred))

    # Plot Actual vs Predicted
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color="blue")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Actual vs Predicted Sales")
    st.pyplot(fig)

    # ----- User Input -----
    st.subheader("🧮 Predict Sales from Custom Ad Spend")

    tv = st.slider("TV Advertising Budget ($)", 0.0, 300.0, 100.0)
    radio = st.slider("Radio Advertising Budget ($)", 0.0, 50.0, 20.0)
    newspaper = st.slider("Newspaper Advertising Budget ($)", 0.0, 120.0, 30.0)

    user_data = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    })

    if st.button("Predict Sales"):
        prediction = model.predict(user_data)[0]
        st.success(f"📊 Predicted Sales: **{prediction:.2f} units**")
