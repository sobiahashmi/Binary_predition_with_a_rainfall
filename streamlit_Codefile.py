# Load Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load Dataset
rainfall = pd.read_csv('train.csv')
rainfall.head()
print("Shape of the Train dataset: ", rainfall.shape)

# Data Preprocessing

# Select features
selected_features = ['id', 'day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed']
X = rainfall[selected_features]
y = rainfall['rainfall']

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=101)
model.fit(X, y)

# Save the trained model to a file  
dump(model, 'random_forest_rainfall.joblib')