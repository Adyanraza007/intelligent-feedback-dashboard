import pandas as pd
import string
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load feedback data
data = pd.read_csv("data/feedback.csv")

# Lists to store sentiment labels
sentiments = []

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# Sentiment analysis
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

# Add sentiment column to dataframe
data["Sentiment"] = sentiments

# Print results
for i in range(len(data)):
    print(f"Feedback: {data['feedback'][i]}")
    print(f"Sentiment: {data['Sentiment'][i]}")
    print("-" * 40)

# Count sentiment values
sentiment_counts = data["Sentiment"].value_counts()

# Pie chart visualization
plt.figure()
plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Student Feedback Sentiment Distribution")
plt.show()
