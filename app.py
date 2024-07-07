# app.py

from flask import Flask, render_template, request
from model import predict_sentiment

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        sentiment = predict_sentiment(review)
        return render_template('index.html', sentiment=sentiment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)