import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_excel("data/student_feedback.xlsx")

df = df[['teaching', 'teaching.1']]
df.columns = ['sentiment', 'feedback']

df = df.dropna()

df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 1 else 0)

X = df['feedback']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

sample = ["The teacher explains concepts clearly"]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)
print("Predicted Sentiment:", "Positive" if prediction[0] == 1 else "Negative")

cv = CountVectorizer(stop_words='english', max_features=20)
X_words = cv.fit_transform(X)

keywords = cv.get_feature_names_out()
word_counts = np.sum(X_words.toarray(), axis=0)

keyword_freq = dict(zip(keywords, word_counts))

sorted_topics = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)

print("\nTop Keywords in Student Feedback:")
print(keywords)

print("\nMost Discussed Topics:")
for topic, count in sorted_topics[:10]:
    print(topic, "→", count)
positive = df[df['sentiment'] == 1].shape[0]
negative = df[df['sentiment'] == 0].shape[0]

print("\nSentiment Summary:")
print("Positive Feedback:", positive)
print("Negative Feedback:", negative)
import matplotlib.pyplot as plt

labels = ['Positive', 'Negative']
sizes = [positive, negative]

plt.figure()
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Student Feedback Sentiment Distribution")
plt.show()
topics = [t[0] for t in sorted_topics[:5]]
counts = [t[1] for t in sorted_topics[:5]]

plt.figure()
plt.bar(topics, counts)
plt.title("Top Issues in Student Feedback")
plt.xlabel("Topics")
plt.ylabel("Frequency")
plt.show()
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved!")
