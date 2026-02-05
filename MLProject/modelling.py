import pandas as pd
import mlflow
import mlflow.sklearn
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup logging
def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )

def train_model():
    setup_logging()
    logging.info("Starting training pipeline...")

    # Muat data
    logging.info("Loading data...")
    try:
        df = pd.read_csv('titanic_preprocessing.csv')
        logging.info("Data loaded from current directory.")
    except FileNotFoundError:
        # Fallback jika dijalankan dari root
        try:
            df = pd.read_csv('Membangun_model/titanic_preprocessing.csv')
            logging.info("Data loaded from Membangun_model directory.")
        except FileNotFoundError:
            logging.error("titanic_preprocessing.csv not found!")
            raise
    
    X = df.drop('Survived', axis=1)
    # Ubah ke float untuk hindari peringatan skema MLflow tentang kolom integer
    X = X.astype(float)
    y = df['Survived']
    
    # Bagi data
    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set eksperimen
    mlflow.set_experiment("Titanic_Basic_Model")
    
    # Aktifkan autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        logging.info("Training model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi (Autolog menangani ini, tapi bagus untuk dicetak)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logging.info(f"Accuracy: {acc}")
        
        # Infer signature (opsional tapi praktik yang baik)
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        logging.info("Model logged to MLflow.")

if __name__ == "__main__":
    train_model()
