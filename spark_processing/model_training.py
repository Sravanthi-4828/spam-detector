from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import *
import findspark
findspark.init()

class SparkMLTrainer:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("SparkMLTraining") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
    
    def load_processed_data(self, data_path):
        """Load processed big data"""
        return self.spark.read.parquet(data_path)
    
    def train_spark_models(self, data_path):
        """Train multiple ML models using Spark MLlib"""
        print("🚀 Training ML models with Spark MLlib...")
        
        # Load processed data
        df = self.load_processed_data(data_path)
        
        # Split data for big data training
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        
        print(f"📊 Training on {train_data.count():,} records")
        print(f"📊 Testing on {test_data.count():,} records")
        
        # Define multiple classifiers
        classifiers = {
            "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label"),
            "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100),
            "Naive Bayes": NaiveBayes(featuresCol="features", labelCol="label"),
            "Linear SVM": LinearSVC(featuresCol="features", labelCol="label")
        }
        
        results = {}
        
        for name, classifier in classifiers.items():
            print(f"\n🔄 Training {name}...")
            
            # Train model
            model = classifier.fit(train_data)
            
            # Make predictions
            predictions = model.transform(test_data)
            
            # Evaluate model
            evaluator_accuracy = MulticlassClassificationEvaluator(
                labelCol="label", 
                predictionCol="prediction", 
                metricName="accuracy"
            )
            
            evaluator_f1 = MulticlassClassificationEvaluator(
                labelCol="label", 
                predictionCol="prediction", 
                metricName="f1"
            )
            
            accuracy = evaluator_accuracy.evaluate(predictions)
            f1_score = evaluator_f1.evaluate(predictions)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1_score,
                'predictions': predictions
            }
            
            print(f"✅ {name} - Accuracy: {accuracy:.4f}, F1-Score: {f1_score:.4f}")
        
        return results, test_data
    
    def cross_validate_model(self, df, classifier, paramGrid):
        """Perform cross-validation for model tuning"""
        evaluator = BinaryClassificationEvaluator(labelCol="label")
        
        crossval = CrossValidator(
            estimator=classifier,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=5,
            parallelism=4  # Use 4 CPU cores
        )
        
        cv_model = crossval.fit(df)
        return cv_model
    
    def ensemble_spark_predictions(self, models_dict, test_data):
        """Create ensemble predictions from multiple Spark models"""
        print("\n🎯 Creating Ensemble Predictions...")
        
        # Get predictions from all models
        all_predictions = []
        for name, result in models_dict.items():
            pred_df = result['predictions'].select("id", "probability", "prediction")
            pred_df = pred_df.withColumnRenamed("prediction", f"pred_{name}")
            all_predictions.append(pred_df)
        
        # Join all predictions
        ensemble_df = test_data
        for pred_df in all_predictions:
            ensemble_df = ensemble_df.join(pred_df, "id", "left")
        
        # Simple voting ensemble
        pred_cols = [f"pred_{name}" for name in models_dict.keys()]
        
        # Calculate majority vote
        vote_expr = sum([col(pred_col) for pred_col in pred_cols])
        ensemble_df = ensemble_df.withColumn("ensemble_vote", vote_expr / len(pred_cols))
        ensemble_df = ensemble_df.withColumn("ensemble_prediction", 
                                           when(col("ensemble_vote") >= 0.5, 1).otherwise(0))
        
        # Evaluate ensemble
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", 
            predictionCol="ensemble_prediction", 
            metricName="accuracy"
        )
        
        ensemble_accuracy = evaluator.evaluate(ensemble_df)
        print(f"✅ Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
        
        return ensemble_df, ensemble_accuracy

if __name__ == "__main__":
    trainer = SparkMLTrainer()
    results, test_data = trainer.train_spark_models("data/processed/emails_processed")
    
    # Create ensemble
    ensemble_df, ensemble_accuracy = trainer.ensemble_spark_predictions(results, test_data)
    
    trainer.spark.stop()