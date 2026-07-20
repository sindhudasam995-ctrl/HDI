from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("model/model.pkl")

# Load the dataset
data = pd.read_csv("hdi_dataset.csv")


# Home Page
@app.route("/")
def home():
    countries = data["Country"].tolist()
    return render_template("index.html", countries=countries)


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    country = request.form["country"]

    # Get selected country's data
    row = data[data["Country"] == country].iloc[0]

    life_expectancy = row["Life_Expectancy"]
    mean_years = row["Mean_Years_of_Schooling"]
    expected_years = row["Expected_Years_of_Schooling"]
    gni = row["GNI_Per_Capita"]

    # Prepare input for the model
    features = [[
        life_expectancy,
        mean_years,
        expected_years,
        gni
    ]]

    # Predict HDI score
    prediction = model.predict(features)[0]
    prediction = round(prediction, 3)

    # HDI Category
    if prediction >= 0.800:
        category = "Very High Human Development"
    elif prediction >= 0.700:
        category = "High Human Development"
    elif prediction >= 0.550:
        category = "Medium Human Development"
    else:
        category = "Low Human Development"

    return render_template(
        "index.html",
        countries=data["Country"].tolist(),
        selected_country=country,
        life_expectancy=life_expectancy,
        mean_years=mean_years,
        expected_years=expected_years,
        gni=gni,
        prediction=prediction,
        category=category
    )


# Run the application
if __name__ == "__main__":
    app.run(debug=True)