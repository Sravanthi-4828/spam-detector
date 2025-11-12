from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import pickle
import findspark
findspark.init()

print("🚀 Starting Big Data Training with PySpark")

# Create Spark session
spark = SparkSession.builder \
    .appName("SpamDetector") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Create sample data
data = []
for i in range(1000):
    if i % 3 == 0:
        data.append((f"Win free prize {i}! Click now!", 1))
    else:
        data.append((f"Meeting scheduled for project {i}", 0))

# Create DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
schema = StructType([
    StructField("text", StringType(), True),
    StructField("label", IntegerType(), True)
])
df = spark.createDataFrame(data, schema)

print(f"📊 Created {df.count()} emails")

# Build pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
idf = IDF(inputCol="raw_features", outputCol="features")

pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf])
model = pipeline.fit(df)
result = model.transform(df)

print("✅ PySpark Processing Completed!")
print("💾 Saving model...")

# Save model
with open('big_data_model.pkl', 'wb') as f:
    pickle.dump({'pipeline': model}, f)

print("🎉 Big Data Model Saved!")
spark.stop()