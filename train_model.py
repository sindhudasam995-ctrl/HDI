import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# Load the Dataset
# -----------------------------

data = pd.read_csv("hdi_dataset.csv")

print("First 5 Rows of the Dataset:")
print(data.head())

# -----------------------------
# Check Missing Values
# -----------------------------

print("\nMissing Values:")
print(data.isnull().sum())

# -----------------------------
# Select Features and Target
# -----------------------------

X = data[[
    "Life_Expectancy",
    "Mean_Years_of_Schooling",
    "Expected_Years_of_Schooling",
    "GNI_Per_Capita"
]]

y = data["HDI_Score"]

# -----------------------------
# Split the Dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train the Linear Regression Model
# -----------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Make Predictions
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Evaluate the Model
# -----------------------------

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Performance")
print("-----------------------------")
print("R2 Score :", r2)
print("Mean Squared Error :", mse)

# -----------------------------
# Save the Model
# -----------------------------

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

print("\nModel saved successfully as model/model.pkl")

# -----------------------------
# Create Images Folder
# -----------------------------

os.makedirs("static/images", exist_ok=True)

# -----------------------------
# HDI Distribution Plot
# -----------------------------

plt.figure(figsize=(6, 4))
data["HDI_Score"].hist()
plt.xlabel("HDI Score")
plt.ylabel("Number of Countries")
plt.title("HDI Score Distribution")
plt.savefig("static/images/hdi_distribution.png")
plt.close()

# -----------------------------
# Scatter Plot
# -----------------------------

plt.figure(figsize=(6, 4))
plt.scatter(
    data["Life_Expectancy"],
    data["HDI_Score"]
)
plt.xlabel("Life Expectancy")
plt.ylabel("HDI Score")
plt.title("Life Expectancy vs HDI Score")
plt.savefig(
    "static/images/life_expectancy_vs_hdi.png"
)
plt.close()

# -----------------------------
# Correlation Heatmap
# -----------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    data.select_dtypes(include=["number"]).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "static/images/correlation_heatmap.png"
)

plt.close()

print("\nGraphs generated successfully!")
print("All files saved inside static/images/")