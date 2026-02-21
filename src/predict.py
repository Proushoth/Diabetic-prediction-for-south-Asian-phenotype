import pickle
import pandas as pd
import os

print("--- Step 1: Script Started ---")

#Load the model
model_path = '../models/random_forest_model.pkl'

if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at {model_path}. Please run train_model.py first.")
else:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("--- Step 2: Model Loaded Successfully ---")

def get_predictions(user_data):
    # This section performs the sri lankan pheotype 
    waist_cm = user_data['waist_inches'] * 2.54
    bmi = user_data['weight'] / ((user_data['height'] / 100) ** 2)
    whtr = waist_cm / user_data['height']
    
    # features that will be fed to the model (must be in the same order as training)
    features = [[
        user_data['age'],
        user_data['gender_encoded'], 
        bmi,
        waist_cm,
        whtr,
        user_data['sitting_encoded'],
        user_data['activity_encoded']
    ]]
    
    prediction = model.predict(features)
    # Get the probability score
    probability = model.predict_proba(features)[0][1]
    
    return "High Risk" if prediction[0] == 1 else "Low Risk", probability

#Test user
print("--- Step 3: Running Test Calculation ---")

test_user = {
    'age': 45,
    'gender_encoded': 1,      # Male
    'height': 170,            # cm
    'weight': 80,             # kg
    'waist_inches': 38,       # Large waist
    'sitting_encoded': 2,     # More than 8 hours
    'activity_encoded': 0     # No exercise
}

try:
    result, score = get_predictions(test_user)
    print("\n" + "="*30)
    print(f"FINAL RESULT: {result}")
    print(f"PROBABILITY: {score:.2%}")
    print("="*30 + "\n")
except Exception as e:
    print(f"An error occurred during prediction: {e}")

print("--- Step 4: Script Finished ---")