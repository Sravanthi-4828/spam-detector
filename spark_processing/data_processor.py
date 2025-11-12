from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline
import findspark
findspark.init()

class BigDataEmailProcessor:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("SpamDetectionBigData") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.executor.memory", "1g") \
            .config("spark.driver.memory", "1g") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("ERROR")
    
    def generate_big_dataset(self, num_records=100000):
        """Generate synthetic dataset - smaller for Python 3.7"""
        print(f"🔄 Generating {num_records:,} email records...")
        
        spam_patterns = [
            "Win free {} now! Click to claim",
            "URGENT: Your {} needs verification",
            "Congratulations! You won {} prize",
            "Get rich quick with {}",
            "Limited time offer: {} free"
        ]
        
        ham_patterns = [
            "Meeting about {} scheduled",
            "Please review the {} document",
            "Project update: {} completed",
            "Team discussion about {}",
            "Reminder: {} deadline approaching"
        ]
        
        spam_words = ["iPhone", "cash", "lottery", "account", "money"]
        ham_words = ["project", "report", "meeting", "deadline", "team"]
        
        records = []
        for i in range(num_records):
            if i % 3 == 0:  # 33% spam
                pattern = spam_patterns[i % len(spam_patterns)]
                word = spam_words[i % len(spam_words)]
                text = pattern.format(word)
                label = 1
            else:  # 67% ham
                pattern = ham_patterns[i % len(ham_patterns)]
                word = ham_words[i % len(ham_words)]
                text = pattern.format(word)
                label = 0
            
            records.append((i, text, label))
        
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("text", StringType(), True),
            StructField("label", IntegerType(), True)
        ])
        
        df = self.spark.createDataFrame(records, schema)
        print(f"✅ Generated {df.count():,} records")
        return df
    
    def preprocess_text_bigdata(self, df):
        """Big Data text preprocessing"""
        print("🔄 Starting Big Data text preprocessing...")
        
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=5000)
        idf = IDF(inputCol="raw_features", outputCol="features")
        
        pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf])
        pipeline_model = pipeline.fit(df)
        processed_df = pipeline_model.transform(df)
        
        print("✅ Text preprocessing completed")
        return processed_df, pipeline_model

if __name__ == "__main__":
    processor = BigDataEmailProcessor()
    big_df = processor.generate_big_dataset(50000)  # Smaller dataset
    processed_df, pipeline_model = processor.preprocess_text_bigdata(big_df)
    processor.spark.stop()