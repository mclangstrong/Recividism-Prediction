import sys
import os

# Add the current directory to the path so we can import app
sys.path.append(os.getcwd())

from app.app import app
from database import db, Prediction

with app.app_context():
    try:
        count = Prediction.query.count()
        print(f"Total Predictions: {count}")
    except Exception as e:
        print(f"Error: {e}")
