# model.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

def train_model():

    df = pd.read_csv('movie_reviews.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)
    

    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    

    X_test_vectorized = vectorizer.transform(X_test)
    accuracy = model.score(X_test_vectorized, y_test)
    print(f"Model accuracy: {accuracy}")

def predict_sentiment(review):
   
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    

    review_vectorized = vectorizer.transform([review])
    
    
    sentiment = model.predict(review_vectorized)[0]
    
    return sentiment

if __name__ == "__main__":
    train_model()
