import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)
pd.set_option('display.max_rows', None)
#print(df)
df = df.replace(0, np.nan)
df = df.fillna(df.median())
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']               # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
joblib.dump(model, 'diabetes_model.pkl')
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
new_input = pd.DataFrame([[5, 116, 74, 0, 0, 25.6, 0.201, 30]], columns=columns)
prediction = model.predict(new_input)
if prediction[0] == 1:
    print("The person is Diabetic.")
else:
    print("The person is NOT Diabetic.")


