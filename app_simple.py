from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
def load_model():
    try:
        with open('spam_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ Model not found! Error: {e}")
        return None

model_data = load_model()

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Spam Detection System</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            textarea { width: 100%; height: 100px; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
            .spam { background: #f8d7da; border: 1px solid #f5c6cb; }
            .ham { background: #d1ecf1; border: 1px solid #bee5eb; }
            .model-result { margin: 5px 0; padding: 5px; background: #f8f9fa; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📧 Spam Detection System</h1>
            <p>Ensemble Machine Learning Approach</p>
            
            <form method="POST" action="/predict">
                <textarea name="text" placeholder="Enter your email or message here..." required></textarea>
                <br>
                <button type="submit">Analyze for Spam</button>
            </form>

            <div style="margin-top: 20px;">
                <h3>Try these examples:</h3>
                <button type="button" onclick="fillText('Win a free iPhone now! Click here to claim your prize.')">Spam Example</button>
                <button type="button" onclick="fillText('Meeting scheduled for tomorrow at 3 PM in conference room.')">Legitimate Example</button>
            </div>
        </div>

        <script>
            function fillText(text) {
                document.querySelector('textarea[name="text"]').value = text;
            }
        </script>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('text', '')
        
        if not text:
            return "Please enter some text to analyze", 400
        
        if model_data is None:
            return "Model not loaded. Please train the model first.", 500
        
        # Vectorize text
        vectorizer = model_data['vectorizer']
        text_vectorized = vectorizer.transform([text])
        
        # Get ensemble prediction
        ensemble_model = model_data['ensemble_model']
        prediction = ensemble_model.predict(text_vectorized)[0]
        probability = ensemble_model.predict_proba(text_vectorized)[0]
        
        # Get individual model predictions
        individual_results = {}
        for name, model in model_data['individual_models'].items():
            try:
                individual_pred = model.predict(text_vectorized)[0]
                individual_prob = model.predict_proba(text_vectorized)[0]
                individual_results[name] = {
                    'prediction': 'SPAM' if individual_pred == 1 else 'LEGITIMATE',
                    'confidence': f"{max(individual_prob)*100:.1f}%",
                    'spam_prob': f"{individual_prob[1]*100:.1f}%"
                }
            except Exception as e:
                individual_results[name] = {'prediction': 'ERROR', 'error': str(e)}
        
        # Create result HTML
        result_html = f"""
        <div class="result {'spam' if prediction == 1 else 'ham'}">
            <h3>{'🚫 SPAM DETECTED' if prediction == 1 else '✅ LEGITIMATE MESSAGE'}</h3>
            <p><strong>Confidence:</strong> {max(probability)*100:.1f}%</p>
            <p><strong>Spam Probability:</strong> {probability[1]*100:.1f}%</p>
            <p><strong>Legitimate Probability:</strong> {probability[0]*100:.1f}%</p>
            
            <h4>Individual Model Predictions:</h4>
        """
        
        for name, model_result in individual_results.items():
            result_html += f"""
            <div class="model-result">
                <strong>{name}:</strong> {model_result['prediction']} (Confidence: {model_result['confidence']})
            </div>
            """
        
        result_html += "</div>"
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Spam Detection Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
                .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .result {{ margin-top: 20px; padding: 15px; border-radius: 5px; }}
                .spam {{ background: #f8d7da; border: 1px solid #f5c6cb; }}
                .ham {{ background: #d1ecf1; border: 1px solid #bee5eb; }}
                .model-result {{ margin: 5px 0; padding: 5px; background: #f8f9fa; border-radius: 3px; }}
                button {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📧 Spam Detection System</h1>
                <p>Ensemble Machine Learning Approach</p>
                
                <form method="POST" action="/predict">
                    <textarea name="text" placeholder="Enter your email or message here...">{text}</textarea>
                    <br>
                    <button type="submit">Analyze for Spam</button>
                </form>
                
                <div style="margin-top: 20px;">
                    <button type="button" onclick="fillText('Win a free iPhone now! Click here to claim your prize.')">Spam Example</button>
                    <button type="button" onclick="fillText('Meeting scheduled for tomorrow at 3 PM in conference room.')">Legitimate Example</button>
                </div>
                
                {result_html}
                
                <div style="margin-top: 20px;">
                    <a href="/"><button>Back to Home</button></a>
                </div>
            </div>
            
            <script>
                function fillText(text) {{
                    document.querySelector('textarea[name="text"]').value = text;
                }}
            </script>
        </body>
        </html>
        """
        
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    print("🚀 Starting Simple Spam Detection Web App...")
    print("📧 Open: http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)