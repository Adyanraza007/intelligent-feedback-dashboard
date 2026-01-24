# ======================================================
# NLP-Based Student Feedback Analysis – FINAL main.py
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from mpl_toolkits.mplot3d import Axes3D


# ======================================================
# 1. Load Dataset
# ======================================================
print("🔥 main.py started")

DATA_PATH = "data/finalDataset0.2.xlsx"
df = pd.read_excel(DATA_PATH)

# Expected columns
df = df[['teaching', 'teaching.1']]
df.columns = ['sentiment', 'feedback']
df.dropna(inplace=True)

df['sentiment'] = df['sentiment'].astype(int)


# ======================================================
# 2. Text Preprocessing
# ======================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df['feedback'] = df['feedback'].apply(clean_text)


# ======================================================
# 3. Features & Labels
# ======================================================
X = df['feedback']
y = df['sentiment']   # 0=Negative, 1=Neutral, 2=Positive


# ======================================================
# 4. Train-Test Split
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ======================================================
# 5. TF-IDF Vectorization
# ======================================================
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=6000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ======================================================
# 6. Model Training
# ======================================================
model = LogisticRegression(
    max_iter=1500,
    class_weight='balanced'
)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    vectorizer.fit_transform(X),
    y,
    cv=5,
    scoring='f1_weighted'
)

print("\n5-Fold Cross Validation F1 Scores:", scores)
print("Mean CV F1 Score:", scores.mean())

model.fit(X_train_vec, y_train)
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]

top_positive_idx = np.argsort(coefs)[-10:]
top_negative_idx = np.argsort(coefs)[:10]

top_positive_words = [feature_names[i] for i in top_positive_idx]
top_negative_words = [feature_names[i] for i in top_negative_idx]

print("\nTop Positive Contributing Words:")
print(top_positive_words)

print("\nTop Negative Contributing Words:")
print(top_negative_words)

plt.figure()
plt.barh(top_positive_words, coefs[top_positive_idx])
plt.title("Top Positive Feature Contributions")
plt.xlabel("Weight")
plt.show()

plt.figure()
plt.barh(top_negative_words, coefs[top_negative_idx])
plt.title("Top Negative Feature Contributions")
plt.xlabel("Weight")
plt.show()



# ======================================================
# 7. Evaluation
# ======================================================
y_pred = model.predict(X_test_vec)

labels = ["Negative", "Neutral", "Positive"]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    model,
    vectorizer.fit_transform(X),
    y,
    cv=5,
    scoring='f1_weighted',
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, label="Training Score")
plt.plot(train_sizes, test_mean, label="Validation Score")
plt.xlabel("Training Size")
plt.ylabel("F1 Score")
plt.title("Learning Curve for Small Dataset Optimization")
plt.legend()
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=labels))


# ======================================================
# 8. Confusion Matrix (FIGURE)
# ======================================================
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix for Sentiment Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


# ======================================================
# 9. 3D Class-wise Metrics (FIGURE)
# ======================================================
report = classification_report(
    y_test,
    y_pred,
    target_names=labels,
    output_dict=True
)

precision = [report[c]["precision"] for c in labels]
recall = [report[c]["recall"] for c in labels]
f1 = [report[c]["f1-score"] for c in labels]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

x = np.arange(len(labels))
dx = dy = 0.4

ax.bar3d(x, np.zeros(3), np.zeros(3), dx, dy, precision)
ax.bar3d(x, np.ones(3), np.zeros(3), dx, dy, recall)
ax.bar3d(x, np.ones(3) * 2, np.zeros(3), dx, dy, f1)

ax.set_xticks(x + dx / 2)
ax.set_xticklabels(labels)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["Precision", "Recall", "F1-score"])
ax.set_zlabel("Score")
ax.set_title("3D Class-wise Performance Metrics")

plt.tight_layout()
plt.savefig("classwise_metrics_3d.png", dpi=300)
plt.show()


# ======================================================
# 10. Keyword / Topic Analysis
# ======================================================
cv = CountVectorizer(
    stop_words='english',
    max_features=30
)

X_words = cv.fit_transform(X)
keywords = cv.get_feature_names_out()
counts = np.sum(X_words.toarray(), axis=0)

topics = dict(zip(keywords, counts))
sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)

print("\nTop Discussion Topics:")
for topic, count in sorted_topics[:10]:
    print(topic, "→", count)


# ======================================================
# 11. Sentiment Distribution (FIGURE)
# ======================================================
sentiment_counts = df['sentiment'].value_counts().sort_index()

plt.figure()
plt.pie(
    sentiment_counts,
    labels=labels,
    autopct='%1.1f%%'
)
plt.title("Student Feedback Sentiment Distribution")
plt.savefig("sentiment_distribution.png", dpi=300)
plt.show()


# ======================================================
# 12. Topic Frequency (FIGURE)
# ======================================================
top_topics = sorted_topics[:5]

plt.figure()
plt.bar(
    [t[0] for t in top_topics],
    [t[1] for t in top_topics]
)
plt.title("Top Issues in Student Feedback")
plt.xlabel("Topics")
plt.ylabel("Frequency")
plt.savefig("keyword_frequency.png", dpi=300)
plt.show()


# ======================================================
# 13. Save Model & Vectorizer
# ======================================================
os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully")
