import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load both the trained model and the scaler
model = pickle.load(open('heart_failure_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. Capture the form data
        feature_values = [
            float(request.form['age']),
            float(request.form['anaemia']),
            float(request.form['creatinine_phosphokinase']),
            float(request.form['diabetes']),
            float(request.form['ejection_fraction']),
            float(request.form['high_blood_pressure']),
            float(request.form['platelets']),
            float(request.form['serum_creatinine']),
            float(request.form['serum_sodium']),
            float(request.form['sex']),
            float(request.form['smoking']),
            float(request.form['time'])
        ]
        
        feature_names = [
            'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 
            'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
        ]

        # Convert to DataFrame with matching feature names for the scaler
        raw_data_df = pd.DataFrame([feature_values], columns=feature_names)
        
        # 2. FIX: Scale the features before predicting so the model gets the right numbers!
        scaled_features = scaler.transform(raw_data_df)
        prediction = model.predict(scaled_features)
        
        # --- GRAPH GENERATION ---
        plt.figure(figsize=(7, 4.5))
        
        # Clear/Display feature names for the chart labels
        display_names = [
            'Age', 'Anaemia', 'Creatinine Phos.', 'Diabetes', 'Ejection Frac.', 
            'High BP', 'Platelets', 'Serum Creat.', 'Serum Sodium', 'Sex', 'Smoking', 'Time'
        ]
        
        # Plotting SCALED values fixes the platelet problem so every bar shows up nicely
        scaled_values_flattened = scaled_features[0]
        
        # Color bars nicely: Blue for negative impact, Red/Orange for positive risk impact
        colors = ['#e74c3c' if val > 0 else '#3498db' for val in scaled_values_flattened]
        
        plt.barh(display_names, scaled_values_flattened, color=colors)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.title('Patient Risk Profile (Relative to Average)')
        plt.xlabel('Variance from Dataset Mean')
        plt.tight_layout()
        
        # Save to static directory
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
            
        graph_path = os.path.join(static_dir, 'result_chart.png')
        plt.savefig(graph_path)
        plt.close()
        
        # Calculate result text based on corrected scaled prediction
        result_text = "Low Risk of Heart Failure" if prediction[0] == 0 else "High Risk of Heart Failure"
        
        return render_template('index.html', 
                               prediction_text=result_text, 
                               graph_url='/static/result_chart.png')

if __name__ == "__main__":
    app.run(debug=True)