import pandas as pd
import string
from textblob import TextBlob
import matplotlib.pyplot as plt
import os


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def analyze_sentiment(data):
    sentiments = []

    for feedback in data["feedback"]:
        cleaned_text = clean_text(feedback)
        analysis = TextBlob(cleaned_text)
        polarity = analysis.sentiment.polarity

        if polarity > 0:
            sentiments.append("Positive")
        elif polarity < 0:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")

    data["Sentiment"] = sentiments
    return data


def visualize_sentiment(data):
    sentiment_counts = data["Sentiment"].value_counts()

    plt.figure()
    plt.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Student Feedback Sentiment Distribution")
    plt.show()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "feedback.csv")

    data = pd.read_csv(DATA_PATH)
    data = analyze_sentiment(data)

    for i in range(len(data)):
        print(f"Feedback: {data['feedback'][i]}")
        print(f"Sentiment: {data['Sentiment'][i]}")
        print("-" * 40)

    visualize_sentiment(data)
