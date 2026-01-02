# Sentiment Analysis with Naive Bayes

## About
This project demonstrates a simple sentiment analysis system using Python. It classifies short movie reviews as `positive` or `negative` using a **Multinomial Naive Bayes** classifier. The project is intended for learning NLP and text classification.

## Features
- Converts text reviews into numeric feature vectors using `CountVectorizer`.
- Splits dataset into training and test sets.
- Trains a Naive Bayes model and evaluates accuracy.
- Predicts sentiment for new reviews.

## Files
- `sentiment_analysis.py` - Main script containing data, model training, and prediction code.
- `README.md` - Project description and usage instructions.

## How to Run
1. Install required packages:
```bash
pip install pandas scikit-learn
