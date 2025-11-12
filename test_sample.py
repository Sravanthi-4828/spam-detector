print("🧪 Testing Python 3.7 Compatibility...")

try:
    import flask
    print("✅ Flask imported successfully")
except ImportError as e:
    print(f"❌ Flask import failed: {e}")

try:
    import sklearn
    print("✅ Scikit-learn imported successfully")
except ImportError as e:
    print(f"❌ Scikit-learn import failed: {e}")

try:
    import pandas
    print("✅ Pandas imported successfully")
except ImportError as e:
    print(f"❌ Pandas import failed: {e}")

try:
    import pyspark
    print("✅ PySpark imported successfully")
except ImportError as e:
    print(f"❌ PySpark import failed: {e}")

print("\n🎯 If all ✅, your environment is ready!")