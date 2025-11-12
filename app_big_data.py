from flask import Flask, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Simple ensemble model for demo
def create_simple_model():
    vectorizer = TfidfVectorizer(max_features=1000)
    models = [
        ('lr', LogisticRegression()),
        ('nb', MultinomialNB())
    ]
    ensemble = VotingClassifier(estimators=models, voting='soft')
    return {'vectorizer': vectorizer, 'ensemble': ensemble}

model_data = create_simple_model()

@app.route('/')
def home():
    return '''
    <html>
    <head><title>Big Data Spam Detection</title></head>
    <body>
        <h1>📧 Big Data Spam Detection</h1>
        <form method="POST" action="/predict">
            <textarea name="text" rows="4" cols="50" placeholder="Enter email text"></textarea><br>
            <button type="submit">Analyze</button>
        </form>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')
    if not text:
        return "Please enter text"
    
    # Simple prediction (replace with actual model)
    spam_keywords = ['win', 'free', 'prize', 'click', 'urgent']
    is_spam = any(keyword in text.lower() for keyword in spam_keywords)
    
    result = "🚫 SPAM" if is_spam else "✅ LEGITIMATE"
    return f'''
    <html>
    <body>
        <h1>Result: {result}</h1>
        <p>Text: {text}</p>
        <a href="/">Back</a>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("🚀 Starting Big Data Spam Detection")
    print("📧 Open: http://localhost:5000")
    app.run(debug=True, port=5000)