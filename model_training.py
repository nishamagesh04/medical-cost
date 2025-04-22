import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset from CSV
df = pd.read_csv('data/insurance.csv')

# Encode categorical variables
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Features and target
X = df.drop(['charges'], axis=1)
y = df['charges']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and columns for app to use
with open('model/medical_cost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("Model trained on CSV data and saved!")
