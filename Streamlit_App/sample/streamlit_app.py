import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Load the trained Random Forest model
model = load('randon_forest_backpack.joblib')

# Create a Streamlit app
st.title("Backpack Price Prediction App")

# Input fields for feature values on the main screen
st.header("Enter Backpack Information")
Brand = st.selectbox("Brand", ('Under Armour','Adidas','Nike','Puma','Jansport'))
Material = st.selectbox("Material", ('Polyester','Leather','Nylon','Canvas'))
Size = st.selectbox("Size", ('Small','Medium','Large'))
Laptop_Compartment = st.selectbox("Laptop Compartment", ('Yes','No'))
if Laptop_Compartment == 'Yes':
   Number_of_Compartment = st.number_input("Number of Compartment", min_value=1, max_value=10, value=1)
else:
   Number_of_Compartment = 0
Waterproof = st.selectbox("Waterproof", ('Yes','No'))
Style = st.selectbox("Style", ('Messenger', 'Tote','Backpack'))
Color = st.selectbox("Color", ('Pink','Gray','Blue','Red','Black','Green'))

# Map input values to numeric using the label mapping
label_mapping = {   # Map correctly with encoded values
    'Under Armour':0,'Adidas':1,'Nike':2,'Puma':3,'Jansport':4,
    'Polyester':0,'Leather':1,'Nylon':2,'Canvas':3,
    'Small':0,'Medium':1,'Large':2,
    'Yes':0,'No':1,
    'Messenger':0, 'Tote':1,'Backpack':2,
    'Pink':0,'Gray':1,'Blue':2,'Red':3,'Black':4,'Green':5
}

Brand = label_mapping[Brand]
Material = label_mapping[Material]
Size = label_mapping[Size]
Laptop_Compartment = label_mapping[Laptop_Compartment]
Waterproof = label_mapping[Waterproof]
Style = label_mapping[Style]
Color = label_mapping[Color]

# Make a prediction using the model
pred = model.predict([[Brand,Material,Size,Laptop_Compartment,Waterproof,Style,Color]])

# Display the prediction result on the main screen
st.header("Prediction Result")
st.write(f"The predicted price of the backpack is: ${pred[0]:.2f}")

# Additional information
st.write("Note: The predicted price is an estimate and may not be accurate.")

