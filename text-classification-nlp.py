import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
    'review': [
        "I love this movie",
        "Horrible acting",
        "What a great film",
        "Worst movie ever",
        "Really enjoyed it",
        "It was terrible",
        "Fantastic performance",
        "Not good at all"
    ],
    'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive',
 'negative']
 }
df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

your_review = ["I hated the plot"]
your_review_vector = vectorizer.transform(your_review)
print("Prediction:", model.predict(your_review_vector))