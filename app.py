import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('heart_failure_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature_names = [
        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
        'ejection_fraction', 'high_blood_pressure', 'platelets',
        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
    ]

    feature_values = [float(request.form[f]) for f in feature_names]
    raw_data_df = pd.DataFrame([feature_values], columns=feature_names)
    scaled_features = scaler.transform(raw_data_df)
    prediction = model.predict(scaled_features)
    proba = model.predict_proba(scaled_features)[0]  # [prob_low, prob_high]
    risk_pct = round(proba[1] * 100, 1)              # probability of high risk

    # ── Dark-themed chart ──
    BG      = '#161b27'
    SURFACE = '#1d2436'
    BORDER  = '#2a3147'
    RED     = '#e05c5c'
    GREEN   = '#3fcf8e'
    TEXT    = '#e8eaf0'
    MUTED   = '#7a84a0'

    display_names = [
        'Age', 'Anaemia', 'Creatinine Phos.', 'Diabetes',
        'Ejection Fraction', 'High Blood Pressure', 'Platelets',
        'Serum Creatinine', 'Serum Sodium', 'Sex', 'Smoking', 'Follow-up Time'
    ]
    scaled_vals = scaled_features[0]
    colors = [RED if v > 0 else GREEN for v in scaled_vals]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    bars = ax.barh(display_names, scaled_vals, color=colors,
                   height=0.6, edgecolor='none')
    ax.axvline(x=0, color=BORDER, linestyle='--', linewidth=1)

    ax.set_title('Patient Risk Profile', color=TEXT, fontsize=12,
                 fontweight='600', pad=12, loc='left')
    ax.set_xlabel('Variance from Dataset Mean', color=MUTED, fontsize=10)

    ax.tick_params(colors=MUTED, labelsize=10)
    ax.spines[:].set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.label.set_color(MUTED)
    for label in ax.get_yticklabels():
        label.set_color(TEXT)
    for label in ax.get_xticklabels():
        label.set_color(MUTED)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=RED, label='Above average (↑ risk)'),
                       Patch(facecolor=GREEN, label='Below average (↓ risk)')]
    ax.legend(handles=legend_elements, loc='lower right',
              facecolor=BG, edgecolor=BORDER,
              labelcolor=MUTED, fontsize=9)

    plt.tight_layout()

    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    plt.savefig(os.path.join(static_dir, 'result_chart.png'),
                dpi=120, facecolor=BG)
    plt.close()

    result_text = "Low Risk of Heart Failure" if prediction[0] == 0 else "High Risk of Heart Failure"
    return render_template('index.html',
                           prediction_text=result_text,
                           risk_pct=risk_pct,
                           graph_url='/static/result_chart.png')

if __name__ == "__main__":
    app.run(debug=True)
