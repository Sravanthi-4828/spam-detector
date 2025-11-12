import pandas as pd
import numpy as np
import os
from pyspark.sql import SparkSession
import findspark
findspark.init()

class EmailDatasetLoader:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("EmailDatasetLoader") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
    
    def load_dataset(self, file_path, text_column='text', label_column='label'):
        """
        Load any email dataset (CSV, JSON, TXT)
        Supports: Enron, SpamAssassin, custom datasets
        """
        print(f"📂 Loading dataset: {file_path}")
        
        # Detect file type and load accordingly
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.txt'):
            df = self._load_text_file(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV, JSON, or TXT")
        
        print(f"✅ Loaded {len(df)} emails")
        
        # Validate required columns
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset")
        
        # If no label column, create dummy labels (for prediction only)
        if label_column not in df.columns:
            print("⚠️  No labels found. Creating dummy labels for training structure.")
            df[label_column] = 0  # Default to ham
        
        # Convert to Spark DataFrame for big data processing
        spark_df = self.spark.createDataFrame(df)
        
        return spark_df, df
    
    def _load_text_file(self, file_path):
        """Load from raw text files (like Enron dataset)"""
        emails = []
        labels = []
        
        # Simple parser for raw email files
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            current_email = []
            for line in f:
                if line.strip() == "---END---" or line.strip() == "**********":
                    if current_email:
                        email_text = ' '.join(current_email)
                        # Simple spam detection based on keywords (for unlabeled data)
                        spam_keywords = ['free', 'win', 'prize', 'cash', 'urgent', 'click', 'subscribe']
                        is_spam = any(keyword in email_text.lower() for keyword in spam_keywords)
                        
                        emails.append(email_text)
                        labels.append(1 if is_spam else 0)
                        current_email = []
                else:
                    current_email.append(line.strip())
        
        return pd.DataFrame({'text': emails, 'label': labels})
    
    def analyze_dataset(self, df):
        """Comprehensive dataset analysis"""
        print("\n📊 DATASET ANALYSIS")
        print("=" * 50)
        
        total_emails = len(df)
        spam_count = df[df['label'] == 1].shape[0] if 'label' in df.columns else "Unknown"
        ham_count = df[df['label'] == 0].shape[0] if 'label' in df.columns else "Unknown"
        
        print(f"Total Emails: {total_emails:,}")
        print(f"Spam Emails: {spam_count}")
        print(f"Legitimate Emails: {ham_count}")
        
        if 'label' in df.columns:
            spam_ratio = spam_count / total_emails * 100
            print(f"Spam Ratio: {spam_ratio:.1f}%")
        
        # Text statistics
        df['text_length'] = df['text'].str.len()
        avg_length = df['text_length'].mean()
        max_length = df['text_length'].max()
        
        print(f"Average Text Length: {avg_length:.0f} characters")
        print(f"Max Text Length: {max_length:,} characters")
        print("=" * 50)
        
        return {
            'total_emails': total_emails,
            'spam_count': spam_count,
            'ham_count': ham_count,
            'avg_text_length': avg_length
        }

# Sample dataset creator for testing
def create_sample_large_dataset(num_emails=10000, save_path='large_email_dataset.csv'):
    """Create a large sample dataset for testing"""
    print(f"🔄 Creating sample dataset with {num_emails:,} emails...")
    
    spam_patterns = [
        "Win a free {}! Click here to claim your {} now! Limited time offer!",
        "URGENT: Your {} account needs verification. Click to secure your {}.",
        "Congratulations! You've been selected for ${} cash prize! Claim now!",
        "Get rich quick with this amazing {} opportunity! No experience needed!",
        "{} alert: Your subscription will expire. Renew now for {} bonus!",
        "Exclusive offer: {}% discount on all products! Shop now!",
        "Your {} reward is waiting! Click to redeem your {} gift!",
        "Warning: {} security breach detected. Verify your {} immediately!",
        "Earn ${} daily from home with this simple {} system!",
        "Limited stock: Get your {} now with free {} delivery!"
    ]
    
    ham_patterns = [
        "Meeting about {} scheduled for {}. Please bring your {} documents.",
        "Hi team, the {} project update: {} has been completed successfully.",
        "Reminder: {} deadline is approaching. Please submit your {} by {}.",
        "Discussion about {} requirements scheduled for {} in {} room.",
        "The {} report is ready for review. Please check the {} section.",
        "Update: {} deployment was successful. Great work on the {} integration!",
        "Please review the {} proposal and provide feedback on the {} aspects.",
        "Team lunch scheduled for {} at {} restaurant. Please confirm availability.",
        "The {} documentation has been updated. Key changes in {} section.",
        "Weekly sync for {} project: Let's discuss {} progress and {} plans."
    ]
    
    spam_words = [['iPhone', 'prize'], ['bank', 'account'], ['1000', 'reward'], 
                  ['investment', 'system'], ['subscription', 'exclusive'], 
                  ['50', 'products'], ['loyalty', 'free'], ['system', 'password'],
                  ['500', 'method'], ['product', 'shipping']]
    
    ham_words = [['quarterly', 'tomorrow', 'financial'], ['software', 'module'],
                 ['project', 'report', 'Friday'], ['client', '3 PM', 'conference'],
                 ['financial', 'budget'], ['server', 'database'], 
                 ['marketing', 'technical'], ['Friday', 'downtown'],
                 ['API', 'authentication'], ['Q3', 'milestone', 'next']]
    
    emails = []
    labels = []
    
    for i in range(num_emails):
        if i % 3 == 0:  # 33% spam
            pattern = spam_patterns[i % len(spam_patterns)]
            words = spam_words[i % len(spam_words)]
            text = pattern.format(*words)
            label = 1
        else:  # 67% ham
            pattern = ham_patterns[i % len(ham_patterns)]
            words = ham_words[i % len(ham_words)]
            text = pattern.format(*words)
            label = 0
        
        emails.append(text)
        labels.append(label)
    
    df = pd.DataFrame({'text': emails, 'label': labels})
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"💾 Sample dataset saved as: {save_path}")
    
    return df

if __name__ == "__main__":
    # Test the dataset loader
    loader = EmailDatasetLoader()
    
    # Create a sample large dataset
    large_df = create_sample_large_dataset(5000)
    
    # Analyze it
    stats = loader.analyze_dataset(large_df)