from flask import Flask, request, render_template
import joblib

# Load saved model
obj = joblib.load('california3.joblib')

model = obj['Model']
columns = obj['Columns']

print("Columns:", columns)

app = Flask(__name__)


# Home page
@app.route('/')
def Main():
    return render_template('index.html', columns=columns)


# Prediction
@app.route('/predict')
def predict():

    Input = []

    # Get values from the UI
    for i in columns:

        val = request.args.get(i, type=float)

        if val is None:
            return f"Missing value for: {i}", 400

        Input.append(val)

    # Make prediction
    prediction = model.predict([Input])

    # Send prediction back to webpage
    return str(prediction[0])


if __name__ == '__main__':
    app.run(debug=True)