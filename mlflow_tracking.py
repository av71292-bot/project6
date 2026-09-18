import mlflow
import mlflow.tensorflow
import joblib
import tensorflow as tf

# 1. Saved LSTM Model aur Scalers load karein
model = tf.keras.models.load_model("lstm_model.keras")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# 2. MLflow Experiment set karein
mlflow.set_experiment("Rossmann_Store_Sales_Prediction")

with mlflow.start_run(run_name="LSTM_Deep_Learning_Model"):
    # Parameters log karein
    mlflow.log_param("model_type", "LSTM")
    mlflow.log_param("time_steps", 7)
    
    # Metrics log karein (Apna actual RMSPE score yahan likhein)
    mlflow.log_metric("rmspe", 33.35)

    # Artifacts log karein
    mlflow.tensorflow.log_model(model, "lstm_model")
    mlflow.log_artifact("scaler_X.pkl")
    mlflow.log_artifact("scaler_y.pkl")

    print("MLflow Tracking for LSTM Model completed successfully!")