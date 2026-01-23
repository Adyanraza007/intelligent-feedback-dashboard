from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import os

# -------------------------
# Flask App Initialization
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# -------------------------
# Load Pretrained BERT Model
# -------------------------
bert_sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("feedback", "")

    result = bert_sentiment(text)[0]

    sentiment = result["label"].capitalize()
    confidence = round(result["score"] * 100, 2)

    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence
    })


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=False)
