# run.py - Run everything with one command!
import os
import subprocess
import sys

def run_command(command, description):
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED: {e}")
        return False

def main():
    print("🤖 STARTING BIG DATA SPAM DETECTION PIPELINE")
    
    # Step 1: Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return
    
    # Step 2: Process data
    if not run_command("python spark_processing/data_processor.py", "Processing Big Data"):
        return
    
    # Step 3: Train Spark models
    if not run_command("python spark_processing/model_training.py", "Training Spark ML Models"):
        return
    
    # Step 4: Train ensemble
    if not run_command("python ensemble_model/ensemble_trainer.py", "Training Advanced Ensemble"):
        return
    
    # Step 5: Start web app
    print(f"\n{'='*50}")
    print("🌐 STARTING WEB APPLICATION")
    print("📧 Open: http://localhost:5000")
    print(f"{'='*50}")
    run_command("python web_app/app.py", "Starting Web Application")

if __name__ == "__main__":
    main()