from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and encoders
model = joblib.load("models/rainwater_model.pkl")
roof_encoder = joblib.load("models/roof_encoder.pkl")
location_encoder = joblib.load("models/location_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    roof_area = float(request.form["roof_area"])
    rainfall = float(request.form["rainfall"])
    roof_type = request.form["roof_type"]
    family_size = int(request.form["family_size"])
    location = request.form["location"]

    # Convert text to numbers
    roof_type_encoded = roof_encoder.transform([roof_type])[0]
    location_encoded = location_encoder.transform([location])[0]

    # Create input
    input_data = pd.DataFrame([[roof_area, rainfall, roof_type_encoded,
                                family_size, location_encoded]],
                              columns=["Roof_Area", "Rainfall",
                                       "Roof_Type", "Family_Size",
                                       "Location"])

    prediction = model.predict(input_data)[0]

    # Tank size recommendation
    if prediction < 30000:
        tank = "5000 Liters"
    elif prediction < 70000:
        tank = "10000 Liters"
    elif prediction < 100000:
        tank = "15000 Liters"
    else:
        tank = "20000 Liters"

    return render_template(
        "result.html",
        water=round(prediction, 2),
        tank=tank
    )


if __name__ == "__main__":
    app.run(debug=True)