import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):

    df.columns = [
        'Timestamp', 'Age', 'Gender', 'Ethnicity', 'Employed', 'Role', 
        'Height', 'Weight', 'Waist_Inches', 'Weight_Increase', 
        'Sitting_Hours', 'Physical_Activity', 'Fast_Food', 'Sugar_Intake', 
        'Family_History', 'Blood_Pressure', 'Target'
    ]
    
    # Phenotype Calculation logic
    df['Waist_cm'] = df['Waist_Inches'] * 2.54
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['WHtR'] = df['Waist_cm'] / df['Height']
    
    # Encode to change words to number for better model accuracy 
    le = LabelEncoder()
    # We use 'astype(str)' to avoid errors with empty values
    cat_cols = ['Gender', 'Sitting_Hours', 'Physical_Activity', 'Target']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    return df