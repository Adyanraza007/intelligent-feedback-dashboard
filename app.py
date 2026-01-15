from flask import Flask, render_template, request, jsonify
import pickle

# Create Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Load trained model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# API for prediction
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data["feedback"]

    # Vectorize input
    vec = vectorizer.transform([text])

    # Predict
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec).max()

    sentiment = "Positive" if prediction == 1 else "Negative"

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(float(probability) * 100, 2)
    })

# Run server
if __name__ == "__main__":
    app.run(debug=True)
