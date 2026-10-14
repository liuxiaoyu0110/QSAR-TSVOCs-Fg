DESCRIPTOR CALCULATOR AND Fg PREDICTOR

This folder contains scripts for calculating molecular descriptors and predicting the gas/particle partitioning fraction (Fg) of tobacco-derived semi-volatile organic compounds (TSVOCs). All files are in the same folder.

Files included:
- descriptors.py : script to compute molecular descriptors from SMILES
- predict_Fg.py           : script to predict Fg using the trained model
- model.pkl               
- scaler.pkl              
- example_input.csv       : example input file for Fg prediction
- requirements.txt        : Python dependencies


REQUIREMENTS

Install the required packages:

pip install -r requirements.txt


1. CALCULATE MOLECULAR DESCRIPTORS

Run the descriptor calculation script with your input file.

Command:
python descriptors.py --input your_data.xlsx --smiles_column SMILES --output descriptors.csv

Input:
- A CSV or Excel file containing a column of SMILES strings.
- Use --smiles_column to specify the column name (default: SMILES).

Output:
- A CSV file (default: descriptors.csv) containing all calculated descriptors.

Optional:
- You can remove constant or near-constant descriptors using the function remove_constant_descriptors in the script.


2. PREDICT Fg

The model expects exactly 9 descriptors in the following order:
1. MID_h
2. Boiling Point
3. Volatility Index
4. MinPartialCharge
5. MATS1c
6. TPSA
7. BCUTp-1l
8. AATS2dv
9. ATS2dv

An example input file example_input.csv is provided. It contains these 9 columns. Column names are ignored; only the order of the first 9 columns matters. You may add extra columns after the first 9; they will be kept in the output but not used for prediction.

To predict, run:
python predict_Fg.py

Then in Python, call:
from predict_Fg import predict_fg
predict_fg('example_input.csv')

Or modify the script to call predict_fg directly with your file path.

Output:
- A CSV file predicted_results.csv with an additional column Predicted_Fg.


INPUT FORMAT FOR PREDICTION

- CSV file.
- First 9 columns must be the descriptors in the exact order listed above.
- Example:

MID_h,Boiling Point,Volatility Index,MinPartialCharge,MATS1c,TPSA,BCUTp-1l,AATS2dv,ATS2dv,Compound_Name,CAS
0.50,200.0,2.50,-0.30,0.10,20.0,0.80,3.20,90.0,Example_1,123-45-6
1.20,165.0,0.80,-0.15,0.05,10.0,1.20,2.90,83.8,Example_2,234-56-7


MODEL SCOPE AND LIMITATIONS

- Trained on 115 tobacco-derived SVOCs measured under a standardized ISO mainstream smoking regime.
- Predictions represent the initial non-equilibrium phase distribution at the emission source, not aged indoor aerosols.
- Applicability domain suggests reliable predictions mainly for polar, heteroatom-containing compounds with TPSA > 10 A^2 and standardized volatility index < 1.0.
- Use with caution for non-polar, highly volatile hydrocarbons.


