from pyspark.ml.linalg import Vectors
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import pickle
from pyspark.sql import SparkSession

class AdvancedEnsembleTrainer:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("AdvancedEnsemble") \
            .getOrCreate()
    
    def convert_spark_to_pandas(self, spark_df, sample_fraction=0.1):
        """Convert Spark DataFrame to pandas for scikit-learn (sampling for large data)"""
        print("🔄 Converting Spark data to pandas (sampling for large datasets)...")
        
        if spark_df.count() > 100000:
            # Sample large dataset for ensemble training
            sampled_df = spark_df.sample(False, sample_fraction, seed=42)
            pandas_df = sampled_df.toPandas()
            print(f"📊 Sampled {len(pandas_df):,} records for ensemble training")
        else:
            pandas_df = spark_df.toPandas()
        
        return pandas_df
    
    def prepare_features_for_ensemble(self, pandas_df):
        """Prepare features for scikit-learn ensemble"""
        # Convert Spark features to numpy arrays
        X = np.array([vec.toArray() for vec in pandas_df['features']])
        y = pandas_df['label'].values
        
        return X, y
    
    def train_advanced_ensemble(self, spark_data_path):
        """Train advanced ensemble model with stacking"""
        print("🚀 Training Advanced Ensemble Model...")
        
        # Load Spark processed data
        spark_df = self.spark.read.parquet(spark_data_path)
        
        # Convert to pandas (sampling for big data)
        pandas_df = self.convert_spark_to_pandas(spark_df, sample_fraction=0.1)
        
        # Prepare features
        X, y = self.prepare_features_for_ensemble(pandas_df)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Ensemble Training: {X_train.shape[0]:,} samples")
        print(f"📊 Ensemble Testing: {X_test.shape[0]:,} samples")
        
        # Define base models
        base_models = [
            ('logistic', LogisticRegression(
                C=1.0, 
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            )),
            ('naive_bayes', MultinomialNB(alpha=0.1)),
            ('svm', SVC(
                C=1.0, 
                kernel='linear', 
                probability=True, 
                random_state=42,
                class_weight='balanced'
            )),
            ('decision_tree', DecisionTreeClassifier(
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            )),
            ('random_forest', RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ))
        ]
        
        # Create voting ensemble
        voting_ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft',  # Use probability voting
            n_jobs=-1
        )
        
        print("🔄 Training Voting Ensemble...")
        voting_ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred = voting_ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Spam']))
        
        # Evaluate individual models
        print("\n📈 Individual Model Performance:")
        individual_results = {}
        for name, model in base_models:
            model.fit(X_train, y_train)
            y_pred_individual = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred_individual)
            individual_results[name] = acc
            print(f"   {name:15}: {acc:.4f}")
        
        # Save ensemble model
        model_data = {
            'ensemble_model': voting_ensemble,
            'individual_models': dict(base_models),
            'individual_accuracies': individual_results,
            'feature_names': None,  # TF-IDF features from Spark
            'ensemble_accuracy': ensemble_accuracy
        }
        
        with open('ensemble_spam_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("💾 Ensemble model saved as 'ensemble_spam_model.pkl'")
        
        return model_data, individual_results, ensemble_accuracy
    
    def big_data_feature_importance(self, spark_df):
        """Analyze feature importance using Spark ML"""
        from pyspark.ml.classification import RandomForestClassifier
        
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(
            featuresCol="features", 
            labelCol="label", 
            numTrees=50,
            seed=42
        )
        
        model = rf.fit(spark_df)
        
        # Get feature importance
        feature_importance = list(zip(
            range(len(model.featureImportances)), 
            model.featureImportances
        ))
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\n🔍 Top 20 Most Important Features:")
        for idx, (feature_idx, importance) in enumerate(feature_importance[:20]):
            print(f"   {idx+1:2}. Feature {feature_idx:4}: {importance:.4f}")
        
        return feature_importance

if __name__ == "__main__":
    ensemble_trainer = AdvancedEnsembleTrainer()
    model_data, individual_results, ensemble_acc = ensemble_trainer.train_advanced_ensemble(
        "data/processed/emails_processed"
    )
    
    # Load Spark data for feature importance analysis
    spark_df = ensemble_trainer.spark.read.parquet("data/processed/emails_processed")
    feature_importance = ensemble_trainer.big_data_feature_importance(spark_df)
    
    ensemble_trainer.spark.stop()