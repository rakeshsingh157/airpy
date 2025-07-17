import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 1: Load the cleaned dataset
df = pd.read_csv('rural_aqi_data.csv')

# Step 2: Features and label
X = df.drop(columns=['RuralAQI'])
y = df['RuralAQI']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("✅ Model Trained Successfully")
print("🔍 Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("🔍 R2 Score:", r2_score(y_test, y_pred))

# Step 6: Save model
joblib.dump(model, 'rural_aqi_model.pkl')
print("💾 Model saved as rural_aqi_model.pkl")
