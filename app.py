from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from nltk.corpus import stopwords
import string
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


app = Flask(__name__)
truenews = pd.read_csv('true.csv')
fakenews = pd.read_csv('fake.csv')
truenews['True/Fake']='True'
fakenews['True/Fake']='Fake'
news = pd.concat([truenews, fakenews])

news["Article"] = news["title"] + news["text"]
news['Clean Text'] = news['Article'].apply(lambda x: word_tokenize(x))
news['Clean Text'] = news['Clean Text'].astype(str)
bow_transformer = CountVectorizer(analyzer=lambda x: word_tokenize(x)).fit(news['Clean Text'])

#news_train, news_test, text_train, text_test = train_test_split(news['Article'], news['True/Fake'], test_size=0.3)

#news['Clean Text'] = news['Article'].apply(lambda x: word_tokenize(str(x)))
#bow_transformer = CountVectorizer(analyzer=lambda x: word_tokenize(x)).fit(news['Clean Text'])
news_bow = bow_transformer.transform(news['Clean Text'])

def tokenize_text(text):
    return word_tokenize(text)

loaded_model = pickle.load(open('model.pkl', 'rb'))
tfidf_transformer = TfidfTransformer().fit(news_bow)
news_tfidf = tfidf_transformer.transform(news_bow)
fakenews_detect_model = MultinomialNB().fit(news_tfidf, news['True/Fake'])

def fake_news_det1(headline):
    input_tokenized = word_tokenize(headline)
    input_data = pd.Series([str(input_tokenized)])
    input_bow = bow_transformer.transform(input_data)
    input_tfidf = tfidf_transformer.transform(input_bow)
    prediction = fakenews_detect_model.predict(input_tfidf)
    return prediction

# Rest of the code remains the same...


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det1(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)