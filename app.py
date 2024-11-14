from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model and required features
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('features.pkl', 'rb') as features_file:
    FEATURE_NAMES = pickle.load(features_file)

app = Flask(__name__)

@app.route('/')
def home():
    # Pass only the required feature names to the template
    return render_template('index.html', feature_names=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs for the required features
        features = [float(request.form[feature]) for feature in FEATURE_NAMES]
        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)[0]
        output = "Parkinson Disease Detected" if prediction == 1 else "No Parkinson Disease"
        return render_template('index.html', prediction_text=f'Result: {output}', success=True, feature_names=FEATURE_NAMES)
    except ValueError:
        return render_template('index.html', prediction_text="Error: Please enter valid numeric values!", success=False, feature_names=FEATURE_NAMES)

if __name__ == "__main__":
    app.run(debug=True)