import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

def predict_fg(input_data_path, model_path='model.pkl', scaler_path='scaler.pkl'):
    """
    Predict Fg values for new compounds.
    
    IMPORTANT: The input CSV must contain exactly 9 columns of molecular descriptors 
    in the following strict order:
    1. MID_h
    2. Boiling Point
    3. Volatility Index
    4. MinPartialCharge
    5. MATS1c
    6. TPSA
    7. BCUTp-1l
    8. AATS2dv
    9. ATS2dv
    """
    print("Loading model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"Loading input data from {input_data_path}...")
    # Read user data
    new_data = pd.read_csv(input_data_path)
    
    # CORE LOGIC: Extract only the first 9 columns of the data matrix, ignoring column names entirely.
    # Assume the user may have added SMILES or CAS numbers before, we remind the user of the exact column positions, or simply assume the user only provided the 9 feature columns.
    
    # Extract the underlying numpy array, discarding the pandas column name binding.
    try:
        
        X_new_raw = new_data.iloc[:, 0:9].values 
    except Exception as e:
        raise ValueError("Error reading data. Please ensure your CSV contains at least 9 columns of descriptor values.")
        
    print("Scaling features...")
    
    X_new_scaled = scaler.transform(X_new_raw)
    
    print("Predicting Fg values...")
    
    raw_predictions = model.predict(X_new_scaled)
    
    
    final_predictions = np.clip(raw_predictions, 0.0, 1.0)
    
    
    new_data['Predicted_Fg'] = final_predictions
    output_path = 'predicted_results.csv'
    new_data.to_csv(output_path, index=False)
    
    print(f"Prediction complete! Results saved to {output_path}.")
    return new_data

if __name__ == "__main__":
    # Example usage:
    # predict_fg('new_compounds_features.csv')
    print("Prediction script ready. Please ensure your input features follow the required order.")