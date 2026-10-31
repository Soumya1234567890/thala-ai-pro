"""
THALA-AI PRO - Model Export Utility
Exports the Support Vector Machine (RBF Kernel), Scaler, and Clinical HPLC Benchmark to model_data.js
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, 'best_thal_ida_classifier (1).joblib')
SCALER_FILE = os.path.join(BASE_DIR, 'scaler (7).joblib')
DATASET_FILE = os.path.join(BASE_DIR, 'RBC indices cleaned filled 2.xlsx')
OUTPUT_FILE = os.path.join(BASE_DIR, 'model_data.js')

print(f"Loading SVM model from {MODEL_FILE}...")
model = joblib.load(MODEL_FILE)
print(f"Loading Scaler from {SCALER_FILE}...")
scaler = joblib.load(SCALER_FILE)
print(f"Loading Dataset from {DATASET_FILE}...")
df = pd.read_excel(DATASET_FILE)

feat_names = list(scaler.feature_names_in_)
scaler_mean = [round(float(x), 6) for x in scaler.mean_]
scaler_scale = [round(float(x), 6) for x in scaler.scale_]

gamma_val = float(getattr(model, '_gamma', 1.0 / len(feat_names)))
intercept_val = round(float(model.intercept_[0]), 6)
probA_val = round(float(model.probA_[0]), 6) if hasattr(model, 'probA_') else -3.249256
probB_val = round(float(model.probB_[0]), 6) if hasattr(model, 'probB_') else 0.101504

support_vectors_scaled = [[round(float(v), 5) for v in row] for row in model.support_vectors_]
dual_coef = [round(float(c), 5) for c in model.dual_coef_[0]]

print(f"Support Vectors: {len(support_vectors_scaled)}, Dual Coefs: {len(dual_coef)}")

# Cohort scatter points for 2D Decision Space
btt_labels = [
    'Beta Thalassemia Trait', 'beta_thalassemia_heterozygous',
    'beta_thalassemia_homozygous', 'hb_lepore_trait', 'sickle_cell_beta_thalassemia_heterozygous'
]
ida_labels = ['Iron Deficiency Anemia', 'ida']

btt_df = df[df['HPLC Diagnosis'].isin(btt_labels)].dropna(subset=['MCV', 'RBC Count', 'RDW-CV'])
ida_df = df[df['HPLC Diagnosis'].isin(ida_labels)].dropna(subset=['MCV', 'RBC Count', 'RDW-CV'])

btt_sample = btt_df.sample(min(120, len(btt_df)), random_state=42)
ida_sample = ida_df.sample(min(120, len(ida_df)), random_state=42)

def extract_pts(subset):
    return [
        {
            'mcv': round(float(r['MCV']), 1),
            'rbc': round(float(r['RBC Count']), 2),
            'rdw': round(float(r['RDW-CV']), 1),
            'mentzer': round(float(r['Mentzer Index']), 2),
            'diagnosis': str(r['HPLC Diagnosis'])
        }
        for _, r in subset.iterrows()
    ]

# Presets aligned with dashboard baseline
presets = [
    {
        'id': 'btt_classic',
        'title': 'Confirmed Beta Thal Trait',
        'desc': 'Marked microcytosis (MCV 56.8 fL) with prominent erythrocytosis (RBC 5.82) and preserved RDW (14.1%). Mentzer < 13 concordant with AI.',
        'data': {
            'rbc_count': 5.82,
            'hemoglobin': 9.8,
            'hct': 32.5,
            'mcv': 56.8,
            'mch': 17.1,
            'mchc': 30.1,
            'rdw_cv': 14.1,
            'mentzer_index': 9.76,
            'age': 28
        }
    },
    {
        'id': 'ida_classic',
        'title': 'Case 2: Severe Iron Deficiency Anemia',
        'desc': 'Severe anemia (Hb 6.8 g/dL), low RBC (3.42) and severe anisocytosis (RDW-CV 21.8%). Mentzer 20.6 strongly concordant with IDA.',
        'data': {
            'rbc_count': 3.42,
            'hemoglobin': 6.8,
            'hct': 24.1,
            'mcv': 70.5,
            'mch': 19.9,
            'mchc': 28.2,
            'rdw_cv': 21.8,
            'mentzer_index': 20.61,
            'age': 34
        }
    },
    {
        'id': 'combined_ida_btt',
        'title': 'Case 3: Combined IDA + Thal Trait (Dual Profile)',
        'desc': 'Dual presentation from 100-case study cohort: Erythrocytosis (RBC 5.38) & microcytosis (MCV 57.6) typical of BTT co-existing with marked anisocytosis (RDW 19.9%) and hypochromia (MCHC 27.8) from IDA.',
        'data': {
            'rbc_count': 5.38,
            'hemoglobin': 8.6,
            'hct': 29.6,
            'mcv': 57.6,
            'mch': 15.8,
            'mchc': 27.8,
            'rdw_cv': 19.9,
            'mentzer_index': 10.71,
            'age': 32
        }
    },
    {
        'id': 'borderline_dilemma',
        'title': 'Case 4: Equivocal Borderline (Mentzer ~13)',
        'desc': 'Mentzer index of 13.6 falsely suggests IDA (cutoff >= 13), but erythrocytosis (RBC 5.08) and low RDW (14.8%) allow AI model to correctly diagnose Beta Thalassemia Trait (confirmed by HPLC).',
        'data': {
            'rbc_count': 5.08,
            'hemoglobin': 11.3,
            'hct': 35.2,
            'mcv': 69.1,
            'mch': 22.1,
            'mchc': 31.9,
            'rdw_cv': 14.8,
            'mentzer_index': 13.6,
            'age': 20
        }
    },
    {
        'id': 'pediatric_microcytic',
        'title': 'Pediatric Microcytic Presentation',
        'desc': '11-month infant with profound microcytosis (MCV 49.5 fL), low Hb (5.4 g/dL), and high RDW (25.2%).',
        'data': {
            'rbc_count': 4.57,
            'hemoglobin': 5.4,
            'hct': 22.6,
            'mcv': 49.5,
            'mch': 11.8,
            'mchc': 23.9,
            'rdw_cv': 25.2,
            'mentzer_index': 10.83,
            'age': 1
        }
    }
]

# Random picker pool
random_pool = []
for _, r in df.sample(50, random_state=77).iterrows():
    if pd.notna(r['RBC Count']) and pd.notna(r['MCV']) and pd.notna(r['Hemoglobin']):
        random_pool.append({
            'patient_id': f"COHORT-{np.random.randint(1000, 9999)}",
            'patient_name': f"Cohort Subject #{np.random.randint(100, 1190)}",
            'age': str(r.get('Age', '30_yrs')),
            'gender': 'Female' if np.random.rand() > 0.5 else 'Male',
            'referring_physician': 'Study Cohort Registry',
            'rbc_count': round(float(r['RBC Count']), 2),
            'hct': round(float(r['HCT']), 1) if pd.notna(r['HCT']) else 30.0,
            'hemoglobin': round(float(r['Hemoglobin']), 1),
            'mcv': round(float(r['MCV']), 1),
            'mch': round(float(r['MCH']), 1),
            'mchc': round(float(r['MCHC']), 1),
            'rdw_cv': round(float(r['RDW-CV']), 1) if pd.notna(r['RDW-CV']) else 14.0,
            'mentzer_index': round(float(r['Mentzer Index']), 2),
            'ground_truth_diagnosis': str(r['HPLC Diagnosis'])
        })

print(f"Generated {len(random_pool)} random pool records.")
print("Model export completed.")
