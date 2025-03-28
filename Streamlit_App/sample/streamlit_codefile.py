# Load Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

# Load Dataset
backpack = pd.read_csv("./playground-series-s5e2/training_extra.csv")
backpack.head()
print("Shape of Training extra:",backpack.shape)

# Data Preprocessing
# Fill missing values and convert to numeric
# backpack['Brand'] = pd.to_numeric(backpack['Brand'],errors='coerce')
# backpack['Material'] = pd.to_numeric(backpack['Material'],errors='coerce')
# backpack['Size'] = pd.to_numeric(backpack['Size'],errors='coerce')
# backpack['Laptop Compartment'] = pd.to_numeric(backpack['Laptop Compartment'],errors='coerce')
# backpack['Waterproof'] = pd.to_numeric(backpack['Waterproof'],errors='coerce')
# backpack['Style'] = pd.to_numeric(backpack['Style'],errors='coerce')
# backpack['Color'] = pd.to_numeric(backpack['Color'],errors='coerce')

backpack['Brand'].fillna(backpack['Brand'].mode().iloc[0],inplace=True)
backpack['Material'].fillna(backpack['Material'].mode()[0],inplace=True)
backpack['Size'].fillna(backpack['Size'].mode()[0],inplace=True)
backpack['Laptop Compartment'].fillna(backpack['Laptop Compartment'].mode()[0],inplace=True)
backpack['Waterproof'].fillna(backpack['Waterproof'].mode()[0],inplace=True)
backpack['Style'].fillna(backpack['Style'].mode()[0],inplace=True)
backpack['Color'].fillna(backpack['Color'].mode()[0],inplace=True)

# Label Encoding (Label/Target)
le = LabelEncoder()
#backpack['Price'] = le.fit_transform(backpack['Price'])

# Encode features
backpack['Brand'] = backpack['Brand'].astype('category').cat.codes
backpack['Material'] = backpack['Material'].astype('category').cat.codes
backpack['Size'] = backpack['Size'].astype('category').cat.codes
backpack['Laptop Compartment']= backpack['Laptop Compartment'].astype('category').cat.codes
backpack['Waterproof'] = backpack['Waterproof'].astype('category').cat.codes
backpack['Style'] = backpack['Style'].astype('category').cat.codes
backpack['Color'] = backpack['Color'].astype('category').cat.codes

# Select features
selected_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
X = backpack[selected_features]
y = backpack['Price']

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=101)
model.fit(X, y)

# Save the trained model to a file  
dump(model, 'randon_forest_backpack.joblib')