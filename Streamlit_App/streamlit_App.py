import streamlit as st
import pandas as pd
from joblib import load

model = load('random_forest_rainfall.joblib')
df = pd.read_csv('../train.csv')
df_test = pd.read_csv('../test.csv')


# Create a Streamlit app
st.title("Rainfall Prediction App")
st.write("This app uses a random forest model to predict rainfall based on temperature and humidity.")

# Number_of_Compartment = st.number_input("Number of Compartment", min_value=1, max_value=10, value=1)
# value = st.slider("Select a value:", min_value=0.0, max_value=100.0, step=0.1)

# Input fields for feature values on the main screen
st.header("Enter Weather Information")

st.header("Display Dataset")
st.dataframe(df.head())
st.header("Summary Statistics")
st.write(df.describe())
st.header("Correlation Matrix")
st.write(df.corr())

st.header("Interactive Filters")
humidity = st.slider("Select Humidity Range", min_value=0, max_value=100, value=(20, 80))
filtered_df = df[(df["humidity"] >= humidity[0]) & (df["humidity"] <= humidity[1])]
st.dataframe(filtered_df)

# Make a prediction using the model
pred = model.predict(df_test)

# Display the prediction result on the main screen
st.header("Prediction Result")
st.write(f"The predicted value of the rainfall is: {pred[0]:.2f}")

