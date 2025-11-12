import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🔄 Training Simple Spam Detection Model...")

# Create sample dataset
data = {
    'text': [
        # Spam messages
        'Win a free iPhone now! Click here to claim your prize.',
        'Congratulations! You won $1000 cash prize. Claim now!',
        'URGENT: Your account will be suspended. Verify now!',
        'Get rich quick with this amazing opportunity!',
        'Free trial! Limited time offer. Sign up now!',
        'You are the winner! Claim your lottery prize!',
        'Bank alert: Your account needs verification.',
        'Earn money from home with no experience!',
        'Special offer: Buy one get one free!',
        'Your credit card has been compromised!',
        
        # Legitimate messages
        'Meeting scheduled for tomorrow at 3 PM in conference room.',
        'Hi John, please review the attached document.',
        'Lunch tomorrow? Let me know your availability.',
        'Project update: The deployment was successful.',
        'Team meeting rescheduled to Friday 2 PM.',
        'Please find the report attached for your review.',
        'Reminder: Project deadline is next Monday.',
        'Can we schedule a call to discuss the requirements?',
        'The quarterly report is ready for your feedback.',
        'Thanks for your help with the project.'
    ],
    'label': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 1=spam, 0=ham
}

df = pd.DataFrame(data)
print(f"📊 Dataset: {len(df)} messages ({sum(df['label'])} spam, {len(df)-sum(df['label'])} legitimate)")

# Prepare features
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Multinomial NB': MultinomialNB(),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Train individual models
print("\n📈 Training Individual Models:")
individual_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    individual_results[name] = accuracy
    print(f"   ✅ {name}: {accuracy:.2%}")

# Create ensemble model
ensemble = VotingClassifier(
    estimators=[
        ('lr', models['Logistic Regression']),
        ('nb', models['Multinomial NB']),
        ('svm', models['SVM']),
        ('rf', models['Random Forest'])
    ],
    voting='soft'
)

ensemble.fit(X_train, y_train)
ensemble_accuracy = ensemble.score(X_test, y_test)
print(f"\n🎯 Ensemble Model Accuracy: {ensemble_accuracy:.2%}")

# Save the model
model_data = {
    'vectorizer': vectorizer,
    'ensemble_model': ensemble,
    'individual_models': models,
    'accuracies': individual_results
}

with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("💾 Model saved as 'spam_model.pkl'")
print("🚀 Model training completed successfully!")