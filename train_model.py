


import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load your dataset
df = pd.read_csv('admission_predict.csv')

# Clean up column names (remove leading and trailing spaces)
df.columns = df.columns.str.strip()

# Check if the columns exist in the dataset
print("Dataset Columns:", df.columns)

# Prepare features (X) and target (y)
X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]
y = df['Chance of Admit']

# Handle any missing values (optional)
# df.dropna(inplace=True)  # If you want to drop rows with missing values

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a pickle file
with open('admission_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model has been trained and saved successfully as 'admission_model.pkl'.")
