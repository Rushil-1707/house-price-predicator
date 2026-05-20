# ─────────────────────────────────────────────
# House Price Prediction - Flask Web App
# Author: Rushil Popat
# ─────────────────────────────────────────────

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from the form
        med_inc    = float(request.form["med_inc"])
        house_age  = float(request.form["house_age"])
        ave_rooms  = float(request.form["ave_rooms"])
        ave_bedrms = float(request.form["ave_bedrms"])
        population = float(request.form["population"])
        ave_occup  = float(request.form["ave_occup"])
        latitude   = float(request.form["latitude"])
        longitude  = float(request.form["longitude"])

        # Prepare input for model
        features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms,
                               population, ave_occup, latitude, longitude]])

        # Scale the input
        features_scaled = scaler.transform(features)

        # Make prediction (price is in $100,000s)
        prediction = model.predict(features_scaled)[0]
        price = round(prediction * 100000, 2)

        return render_template("index.html",
                               prediction=f"${price:,.2f}")

    except Exception as e:
        return render_template("index.html",
                               prediction=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
