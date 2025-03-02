import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('soil_data.csv')
X = df.drop('soil_quality', axis=1)
y = df['soil_quality']
X.fillna(X.mean(), inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
new_soil_data = pd.DataFrame({'feature1': [10], 'feature2': [20], 'feature3': [30]})  # Replace with your feature names and values
new_soil_data.fillna(X.mean(), inplace=True) # Fill missing values in new data
predicted_soil_quality = model.predict(new_soil_data)
print(f"Predicted Soil Quality: {predicted_soil_quality[0]}")
