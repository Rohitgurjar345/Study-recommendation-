import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
import os

# Load the dataset
data = pd.read_csv(r"C:\Users\Rohit\Desktop\Gitdemo\sankalp\Education-Recommendation-System-Student-Subject-and-Studies-Recommendation-system-machine-learning\student-scores.csv")

# Drop irrelevant columns
data.drop(columns=['id', 'first_name', 'last_name', 'email'], inplace=True)

# Feature engineering
data['total_score'] = data[['math_score', 'history_score', 'physics_score', 'chemistry_score', 
                            'biology_score', 'english_score', 'geography_score']].sum(axis=1)
data['average_score'] = data['total_score'] / 7

# Encode categorical variables
data['gender'] = data['gender'].map({'male': 0, 'female': 1})
data['part_time_job'] = data['part_time_job'].map({False: 0, True: 1})
data['extracurricular_activities'] = data['extracurricular_activities'].map({False: 0, True: 1})
career_mapping = {
    'Lawyer': 0, 'Doctor': 1, 'Government Officer': 2, 'Artist': 3, 'Unknown': 4,
    'Software Engineer': 5, 'Teacher': 6, 'Business Owner': 7, 'Scientist': 8,
    'Banker': 9, 'Writer': 10, 'Accountant': 11, 'Designer': 12, 'Construction Engineer': 13,
    'Game Developer': 14, 'Stock Investor': 15, 'Real Estate Developer': 16
}
data['career_aspiration'] = data['career_aspiration'].map(career_mapping)

# Define features and target
X = data.drop(columns=['career_aspiration'])
y = data['career_aspiration']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train_balanced)

# Evaluate the model (optional)
train_score = model.score(X_train_scaled, y_train_balanced)
test_score = model.score(X_test_scaled, y_test)
print(f"Training Accuracy: {train_score:.2f}")
print(f"Testing Accuracy: {test_score:.2f}")

# Save the scaler and model to the "Models" directory
os.makedirs("Models", exist_ok=True)
with open("Models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("Models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Scaler and model saved successfully to the 'Models' directory.")