from flask import Flask, request, render_template
from flask_compress import Compress
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import os
df = pd.DataFrame()
directory = r"C:\Users\dell\Downloads\PROJECT\HPP\HPP\artifacts"
if not os.path.exists(directory):
    os.makedirs(directory)

ll = pd.read_csv(os.path.join(directory, "locations.csv"))
__locations = ll['locations'].to_list()

app = Flask(__name__)
Compress(app)

model = pickle.load(open("C:/Users/dell/Downloads/PROJECT/HPP/HPP/artifacts/house_prices_model.pickle", 'rb'))
model_path = Path(__file__).parent / "artifacts" / "house_prices_model.pickle"
model = pickle.load(open(model_path, 'rb'))


@app.route('/')
def home():
    locs = ll['locations'][1:]
    return render_template('index.html', locs=locs)


def get_estimated_price(sqft, location, bhk, bath):
    loc_index = __locations.index(location.lower())+3
    x = np.zeros(len(__locations)+3)
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return round(model.predict([x])[0], 2)


@app.route('/predict', methods=['POST'])
def predict():
    total_sqft = request.form.get('total_sqft')
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    loc = request.form.get('loc')
    if (loc == "Choose a Location"):
        return "Choose a Location"
    if (str(total_sqft) == ""):
        return "Enter Area"
    if int(total_sqft) < 450:
        return "Minimum Area: 450 sft"
    if int(total_sqft) > 8000:
        return "Maximum Area: 8000 sft"
    prediction = "₹ "
    prediction += str(get_estimated_price(total_sqft,
                                          loc, bhk, bath)) + " Lakh"
    return prediction


if __name__ == "__main__":
    app.run(debug=True)