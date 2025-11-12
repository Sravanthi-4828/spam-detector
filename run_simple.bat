@echo off
echo Starting Simple Spam Detection System...
echo.

echo Step 1: Checking Python version...
python --version

echo.
echo Step 2: Training the model...
python train_simple.py

echo.
echo Step 3: Starting web application...
echo 📧 Open: http://localhost:5000
echo.
python app_simple.py

pause