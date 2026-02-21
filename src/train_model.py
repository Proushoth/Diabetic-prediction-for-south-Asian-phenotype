import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from preprocess import clean_data

# Load the data
df = pd.read_csv(r'C:\ProKo\cobsccomp4y241p-025_15386777_NB6012CEM_FinalResearch\Diabetic_prediction_system\data\South_Asian_Phenotype_Data_300.csv')

# Clean it using our preprocess script
df = clean_data(df)

# Select Features for the AI
features = ['Age', 'Gender', 'BMI', 'Waist_cm', 'WHtR', 'Sitting_Hours', 'Physical_Activity']
X = df[features]
y = df['Target']

# Initialize and Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# SAVE THE MODEL (The "Pickle" file)
with open('../models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Project Started: Model trained and saved in /models/ folder!")