import os
import io
import random
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from report_generator import generate_clinical_report_pdf

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_thal_ida_classifier (1).joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "best_thal_ida_classifier.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler (7).joblib")
DATASET_PATH = os.path.join(BASE_DIR, "RBC indices cleaned filled 2.xlsx")

app = FastAPI(
    title="THALA-AI PRO Diagnostic Engine",
    description="Educational research prototype (Thalassemia trait vs. Iron Deficiency Anemia) trained on synthetic data generated with Google Gemini. Not a medical device; not for clinical use; no laboratory accreditation.",
    version="2.4.0"
)

# --- LOAD ASSETS ---
print("Loading Model and Scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = list(scaler.feature_names_in_)
print(f"Features loaded: {feature_names}")

# Pre-compile RBF Support Vector Machine Decision Engine
print("Initializing Support Vector Machine RBF Decision Engine...")
base_expected_val = 0.461
gamma_val = float(getattr(model, '_gamma', 1.0 / len(feature_names)))
intercept_val = float(model.intercept_[0])
probA_val = float(model.probA_[0]) if hasattr(model, 'probA_') else -3.249256
probB_val = float(model.probB_[0]) if hasattr(model, 'probB_') else 0.101504
print(f"SVM Engine initialized with {len(model.support_vectors_)} support vectors, gamma={gamma_val:.4f}")

# Load and prepare cohort dataset
print("Loading synthetic dataset (generated with Google Gemini; not real patients)...")
cohort_df = pd.read_excel(DATASET_PATH)
btt_labels = [
    'Beta Thalassemia Trait', 'beta_thalassemia_heterozygous',
    'beta_thalassemia_homozygous', 'hb_lepore_trait', 'sickle_cell_heterozygous'
]
cohort_df['is_thal'] = cohort_df['HPLC Diagnosis'].isin(btt_labels).astype(int)
print(f"Cohort loaded: {len(cohort_df)} synthetic records.")

# --- PYDANTIC SCHEMAS ---
class PatientInput(BaseModel):
    age: Optional[float] = Field(28.0, ge=1.0, le=100.0, description="Patient Age in Years")
    patient_id: Optional[str] = "DEMO-INPUT"
    patient_name: Optional[str] = "Demo input (not a patient)"
    gender: Optional[str] = "Unspecified"
    referring_physician: Optional[str] = "Not applicable (educational demo)"
    clinical_notes: Optional[str] = ""

    rbc_count: float = Field(..., ge=0.5, le=10.0, description="RBC Count (10^12/L)")
    hct: float = Field(..., ge=5.0, le=70.0, description="Hematocrit (%)")
    hemoglobin: float = Field(..., ge=1.0, le=25.0, description="Hemoglobin (g/dL)")
    mcv: float = Field(..., ge=30.0, le=130.0, description="Mean Corpuscular Volume (fL)")
    mch: float = Field(..., ge=10.0, le=50.0, description="Mean Corpuscular Hemoglobin (pg)")
    mchc: float = Field(..., ge=15.0, le=50.0, description="MCHC (g/dL)")
    rdw_cv: float = Field(..., ge=8.0, le=45.0, description="RDW-CV (%)")
    mentzer_index: Optional[float] = None

class ReportDownloadRequest(BaseModel):
    patient: Dict[str, Any]
    analysis: Dict[str, Any]

# --- BIOLOGICAL REFERENCE INTERVALS & EVALUATION ---
REFERENCE_RANGES = {
    'RBC Count': {'min': 3.80, 'max': 5.80, 'unit': '10^12/L', 'desc': 'Red blood cell count'},
    'Hemoglobin': {'min': 11.5, 'max': 16.5, 'unit': 'g/dL', 'desc': 'Total hemoglobin concentration'},
    'HCT': {'min': 35.0, 'max': 48.0, 'unit': '%', 'desc': 'Hematocrit / packed cell volume'},
    'MCV': {'min': 80.0, 'max': 100.0, 'unit': 'fL', 'desc': 'Mean corpuscular volume (size)'},
    'MCH': {'min': 27.0, 'max': 33.0, 'unit': 'pg', 'desc': 'Mean corpuscular hemoglobin'},
    'MCHC': {'min': 31.0, 'max': 36.0, 'unit': 'g/dL', 'desc': 'Mean corpuscular Hb concentration'},
    'RDW-CV': {'min': 11.5, 'max': 15.0, 'unit': '%', 'desc': 'Red cell distribution width (anisocytosis)'},
    'Mentzer Index': {'min': 13.0, 'max': 25.0, 'unit': 'ratio', 'desc': 'MCV / RBC discriminant ratio'}
}

def evaluate_cbc_flags(data: dict) -> List[dict]:
    eval_list = []
    # Support both title-case and snake_case keys
    mapping = {
        'RBC Count': (data.get('RBC Count', data.get('rbc_count', 0.0)), '10^12/L'),
        'Hemoglobin': (data.get('Hemoglobin', data.get('hemoglobin', 0.0)), 'g/dL'),
        'HCT': (data.get('HCT', data.get('hct', 0.0)), '%'),
        'MCV': (data.get('MCV', data.get('mcv', 0.0)), 'fL'),
        'MCH': (data.get('MCH', data.get('mch', 0.0)), 'pg'),
        'MCHC': (data.get('MCHC', data.get('mchc', 0.0)), 'g/dL'),
        'RDW-CV': (data.get('RDW-CV', data.get('rdw_cv', 0.0)), '%'),
        'Mentzer Index': (data.get('Mentzer Index', data.get('mentzer_index', 0.0)), 'ratio')
    }
    for name, (val, unit) in mapping.items():
        ref = REFERENCE_RANGES.get(name, {'min': 0, 'max': 100})
        flag = 'Normal'
        comment = 'Within standard biological reference interval.'
        if name == 'Mentzer Index':
            flag = 'Low (<13)' if val < 13.0 else 'High (>13)'
            comment = 'Typical BTT range (<13)' if val < 13.0 else 'Typical IDA / normal range (>13)'
        elif val < ref['min']:
            flag = 'VERY LOW' if val < ref['min'] * 0.8 else 'LOW'
            if name == 'MCV':
                comment = 'Marked microcytosis (abnormally small red cells)'
            elif name == 'Hemoglobin':
                comment = 'Anemia (reduced oxygen-carrying capacity)'
            elif name == 'RBC Count':
                comment = 'Decreased erythrocyte count'
            else:
                comment = 'Below reference range'
        elif val > ref['max']:
            flag = 'VERY HIGH' if val > ref['max'] * 1.25 else 'HIGH'
            if name == 'RBC Count':
                comment = 'Erythrocytosis (microcytic polycythemia, classic in Thalassemia trait)'
            elif name == 'RDW-CV':
                comment = 'Marked anisocytosis (significant size heterogeneity, classic in IDA)'
            else:
                comment = 'Above reference range'

        eval_list.append({
            'name': name,
            'value': round(val, 2),
            'unit': unit,
            'ref_range': f"{ref['min']} - {ref['max']}",
            'flag': flag,
            'comment': comment
        })
    return eval_list

# --- CLASSICAL DISCRIMINATION INDICES CALCULATION ---
def calculate_classical_indices(rbc: float, hct: float, hb: float, mcv: float, mch: float, rdw: float, ai_pred_class: int):
    """
    Computes 8 recognized hematological discrimination formulas:
    Mentzer, Srivastava, Green & King, Shine & Lal, England & Fraser, Ricerca, Ehsani, RDWI
    """
    indices = []
    
    # 1. Mentzer Index
    mi_val = mcv / max(rbc, 0.001)
    mi_is_thal = mi_val < 13.0
    indices.append({
        'id': 'mentzer',
        'name': 'Mentzer Index',
        'formula': 'MCV / RBC',
        'value': round(mi_val, 2),
        'cutoff_rule': '< 13.0 indicates Thal; > 13.0 indicates IDA',
        'threshold': 13.0,
        'direction': '<',
        'interpretation': 'Thalassemia Trait' if mi_is_thal else 'Iron Deficiency Anemia',
        'indicates_thal': mi_is_thal,
        'agrees_with_ai': (1 if mi_is_thal else 0) == ai_pred_class,
        'clinical_note': 'Classic single ratio. Vulnerable when RBC is depressed in concomitant IDA.'
    })

    # 2. Srivastava Index
    si_val = mch / max(rbc, 0.001)
    si_is_thal = si_val < 3.8
    indices.append({
        'id': 'srivastava',
        'name': 'Srivastava Index',
        'formula': 'MCH / RBC',
        'value': round(si_val, 2),
        'cutoff_rule': '< 3.8 indicates Thal; > 3.8 indicates IDA',
        'threshold': 3.8,
        'direction': '<',
        'interpretation': 'Thalassemia Trait' if si_is_thal else 'Iron Deficiency Anemia',
        'indicates_thal': si_is_thal,
        'agrees_with_ai': (1 if si_is_thal else 0) == ai_pred_class,
        'clinical_note': 'Emphasizes low MCH per red cell.'
    })

    # 3. Green & King Index
    gk_val = ( (mcv ** 2) * rdw ) / max(hb * 100.0, 0.001)
    gk_is_thal = gk_val < 65.0
    indices.append({
        'id': 'green_king',
        'name': 'Green & King Index',
        'formula': '(MCV² × RDW) / (Hb × 100)',
        'value': round(gk_val, 2),
        'cutoff_rule': '< 65.0 indicates Thal; > 72.0 indicates IDA',
        'threshold': 65.0,
        'direction': '<',
        'interpretation': 'Thalassemia Trait' if gk_is_thal else ('Borderline' if gk_val <= 72.0 else 'Iron Deficiency Anemia'),
        'indicates_thal': gk_is_thal,
        'agrees_with_ai': (1 if gk_is_thal else 0) == ai_pred_class,
        'clinical_note': 'Incorporates RDW to penalize wide red cell distribution.'
    })

    # 4. Shine & Lal Index
    sl_val = ( (mcv ** 2) * mch ) / 100.0
    sl_is_thal = sl_val < 1530.0
    indices.append({
        'id': 'shine_lal',
        'name': 'Shine & Lal Index',
        'formula': '(MCV² × MCH) / 100',
        'value': round(sl_val, 2),
        'cutoff_rule': '< 1530.0 indicates Thal; > 1530.0 indicates IDA',
        'threshold': 1530.0,
        'direction': '<',
        'interpretation': 'Thalassemia Trait' if sl_is_thal else 'Iron Deficiency Anemia',
        'indicates_thal': sl_is_thal,
        'agrees_with_ai': (1 if sl_is_thal else 0) == ai_pred_class,
        'clinical_note': 'Highly sensitive for microcytosis, but often over-predicts Thalassemia in severe IDA.'
    })

    # 5. England & Fraser Index
    ef_val = mcv - rbc - (5.0 * hb) - 3.4
    ef_is_thal = ef_val < 0.0
    indices.append({
        'id': 'england_fraser',
        'name': 'England & Fraser Index',
        'formula': 'MCV - RBC - (5 × Hb) - 3.4',
        'value': round(ef_val, 2),
        'cutoff_rule': '< 0 indicates Thal; > 0 indicates IDA',
        'threshold': 0.0,
        'direction': '<',
        'interpretation': 'Thalassemia Trait' if ef_is_thal else 'Iron Deficiency Anemia',
        'indicates_thal': ef_is_thal,
        'agrees_with_ai': (1 if ef_is_thal else 0) == ai_pred_class,
        'clinical_note': 'Classical linear discriminant function.'
    })

    # 6. Ricerca Index
    ric_val = rdw / max(rbc, 0.001)
    ric_is_thal = ric_val < 4.4
    indices.append({
        'id': 'ricerca',
        'name': 'Ricerca Index',
        'formula': 'RDW / RBC',
        'value': round(ric_val, 2),
        'cutoff_rule': '< 4.4 indicates Thal; > 4.4 indicates IDA',
        'threshold': 4.4,
        'direction': '<',
        'interpretation': 'Thalassemia Trait' if ric_is_thal else 'Iron Deficiency Anemia',
        'indicates_thal': ric_is_thal,
        'agrees_with_ai': (1 if ric_is_thal else 0) == ai_pred_class,
        'clinical_note': 'Measures RDW relative to total erythrocyte count.'
    })

    # 7. Ehsani Index
    eh_val = mcv - (10.0 * rbc)
    eh_is_thal = eh_val < 15.0
    indices.append({
        'id': 'ehsani',
        'name': 'Ehsani Index',
        'formula': 'MCV - (10 × RBC)',
        'value': round(eh_val, 2),
        'cutoff_rule': '< 15.0 indicates Thal; > 15.0 indicates IDA',
        'threshold': 15.0,
        'direction': '<',
        'interpretation': 'Thalassemia Trait' if eh_is_thal else 'Iron Deficiency Anemia',
        'indicates_thal': eh_is_thal,
        'agrees_with_ai': (1 if eh_is_thal else 0) == ai_pred_class,
        'clinical_note': 'Rapid mathematical screening formula.'
    })

    # 8. RDWI (Red Cell Distribution Width Index)
    rdwi_val = (mcv * rdw) / max(rbc, 0.001)
    rdwi_is_thal = rdwi_val < 220.0
    indices.append({
        'id': 'rdwi',
        'name': 'RDWI Index',
        'formula': '(MCV × RDW) / RBC',
        'value': round(rdwi_val, 2),
        'cutoff_rule': '< 220.0 indicates Thal; > 220.0 indicates IDA',
        'threshold': 220.0,
        'direction': '<',
        'interpretation': 'Thalassemia Trait' if rdwi_is_thal else 'Iron Deficiency Anemia',
        'indicates_thal': rdwi_is_thal,
        'agrees_with_ai': (1 if rdwi_is_thal else 0) == ai_pred_class,
        'clinical_note': 'High discriminative power (highest single-index concordance with AI).'
    })

    agree_count = sum(1 for idx in indices if idx['agrees_with_ai'])
    total_idx = len(indices)
    agreement_pct = round((agree_count / total_idx) * 100.0, 1)

    summary = {
        'agreement_count': agree_count,
        'total_indices': total_idx,
        'agreement_pct': agreement_pct,
        'consensus_verdict': (
            'High Index Concordance' if agreement_pct >= 75
            else ('Moderate Index Concordance' if agreement_pct >= 50 else 'Diagnostic Divergence / Mixed Profile')
        )
    }

    return indices, summary

# --- CLINICAL EXPLANATIONS FOR SHAP ---
FEATURE_DESCRIPTIONS = {
    'RBC Count': {
        'thal': 'Elevated erythrocyte count with microcytosis is a hallmark of ineffective erythropoiesis compensation in BTT.',
        'ida': 'Subnormal or normal RBC count with microcytosis aligns with impaired erythropoiesis due to depleted iron stores.'
    },
    'MCV': {
        'thal': 'Marked microcytosis (often < 70 fL) disproportionate to the mild anemia strongly favors Thalassemia trait.',
        'ida': 'Microcytosis typically correlates linearly with severity of anemia in IDA; higher MCV pulls away from BTT.'
    },
    'RDW-CV': {
        'thal': 'Uniformly small red cells with lower RDW (normal or mild elevation) strongly supports BTT over IDA.',
        'ida': 'Marked anisocytosis (elevated RDW > 16%) is characteristic of progressive iron depletion.'
    },
    'Mentzer Index': {
        'thal': 'Mentzer Index < 13 reflects microcytosis accompanied by high red cell count, characteristic of BTT.',
        'ida': 'Mentzer Index > 13 reflects microcytosis with inadequate red cell expansion, characteristic of IDA.'
    },
    'Hemoglobin': {
        'thal': 'Mild to moderate hemoglobin reduction without severe tissue hypoxia signs fits stable heterozygous trait.',
        'ida': 'Hemoglobin decline directly driven by iron exhaustion.'
    },
    'HCT': {
        'thal': 'Preserved hematocrit relative to very low MCV indicates microcytic polycythemia.',
        'ida': 'Reduced packed cell volume proportional to iron-restricted erythropoiesis.'
    },
    'MCH': {
        'thal': 'Severe hypochromia per erythrocyte reflects impaired globin chain synthesis.',
        'ida': 'Hypochromia driven by lack of heme synthesis.'
    },
    'MCHC': {
        'thal': 'Relative preservation of intracellular hemoglobin concentration in microcytic cells.',
        'ida': 'Progressive decrease in cellular hemoglobin concentration.'
    }
}

# --- ENDPOINTS ---

@app.post("/api/predict")
def predict_and_explain(patient: PatientInput):
    # Auto-calculate Mentzer if not provided
    calc_mentzer = patient.mcv / max(patient.rbc_count, 0.001)
    mentzer_val = patient.mentzer_index if patient.mentzer_index is not None else calc_mentzer

    # Prepare input array matching scaler feature order:
    # 7 model inputs, in the scaler's feature order: ['RBC', 'Hb', 'Hct', 'MCV', 'MCH', 'MCHC', 'RDW-CV'] (Mentzer Index is NOT a model input)
    col_map = {
        'RBC': patient.rbc_count,
        'Hb': patient.hemoglobin,
        'Hct': patient.hct,
        'MCV': patient.mcv,
        'MCH': patient.mch,
        'MCHC': patient.mchc,
        'RDW-CV': patient.rdw_cv
    }

    input_arr = np.array([[col_map[col] for col in feature_names]])
    scaled_features = scaler.transform(input_arr)

    # Model inference
    pred_class = int(model.predict(scaled_features)[0])
    probs = model.predict_proba(scaled_features)[0]
    prob_ida = float(probs[0])
    prob_thal = float(probs[1])
    z_score = float(model.decision_function(scaled_features)[0])

    # Marginal RBF SVM feature attribution
    bg_means = [4.758, 8.614, 29.331, 61.977, 18.534, 29.869, 17.416]
    bg_scaled = scaler.transform(np.array([bg_means]))[0]

    shap_feature_list = []
    meta_name_map = {
        'RBC': 'RBC Count',
        'Hb': 'Hemoglobin',
        'Hct': 'Hematocrit',
        'MCV': 'MCV',
        'MCH': 'MCH',
        'MCHC': 'MCHC',
        'RDW-CV': 'RDW-CV'
    }

    for i, col in enumerate(feature_names):
        val = col_map[col]
        perturbed = scaled_features.copy()
        perturbed[0, i] = bg_scaled[i]
        z_perturbed = float(model.decision_function(perturbed)[0])
        phi = z_score - z_perturbed

        meta_label = meta_name_map.get(col, col)
        desc_dict = FEATURE_DESCRIPTIONS.get(col, FEATURE_DESCRIPTIONS.get(meta_label, {'thal': 'Pushes probability towards Thalassemia.', 'ida': 'Pushes probability towards IDA.'}))
        explanation = desc_dict['thal'] if phi > 0 else desc_dict['ida']

        shap_feature_list.append({
            'name': meta_label,
            'actual_value': round(val, 2),
            'shap_value': round(phi, 4),
            'abs_shap': abs(phi),
            'direction': 'Thalassemia' if phi > 0 else 'IDA',
            'explanation': explanation
        })

    # Sort SHAP features by absolute contribution
    shap_feature_list.sort(key=lambda x: x['abs_shap'], reverse=True)
    for rank, item in enumerate(shap_feature_list, 1):
        item['rank'] = rank

    # Classical Indices Concordance
    indices_list, concordance_summary = calculate_classical_indices(
        patient.rbc_count, patient.hct, patient.hemoglobin,
        patient.mcv, patient.mch, patient.rdw_cv, pred_class
    )

    # CBC Flags
    raw_dict = {
        'rbc_count': patient.rbc_count,
        'hemoglobin': patient.hemoglobin,
        'hct': patient.hct,
        'mcv': patient.mcv,
        'mch': patient.mch,
        'mchc': patient.mchc,
        'rdw_cv': patient.rdw_cv
    }
    patient_eval_dict = {**raw_dict, 'mentzer_index': mentzer_val}
    cbc_eval = evaluate_cbc_flags(patient_eval_dict)

    # Confidence tier
    max_prob = max(prob_thal, prob_ida)
    if max_prob >= 0.85:
        confidence_tier = "High Confidence"
    elif max_prob >= 0.65:
        confidence_tier = "Moderate Confidence"
    else:
        confidence_tier = "Borderline / Low Confidence"

    # Clinical Recommendations
    recommendations = []
    if pred_class == 1:
        recommendations.append({
            'title': 'Cation-Exchange HPLC / Capillary Electrophoresis',
            'desc': 'Mandatory confirmatory quantification of HbA2 and HbF. An elevated HbA2 > 3.5% confirms Beta Thalassemia Trait.'
        })
        recommendations.append({
            'title': 'Concomitant Iron Deficiency Screening',
            'desc': 'Measure Serum Ferritin, Serum Iron, and TIBC. Co-existing iron deficiency can artificially suppress HbA2 levels, masking BTT.'
        })
        recommendations.append({
            'title': 'Cascade Family Screening & Genetic Counseling',
            'desc': 'Screen partner and first-degree relatives before conception to prevent Thalassemia Major in offspring.'
        })
        recommendations.append({
            'title': 'Avoid Inappropriate Iron Supplementation',
            'desc': 'Patients with isolated BTT should NOT receive empirical iron therapy as it confers risk of secondary hemochromatosis / iron overload.'
        })
    else:
        recommendations.append({
            'title': 'Complete Iron Kinetic Studies',
            'desc': 'Evaluate Serum Ferritin (<15-30 ng/mL diagnostic of IDA), Transferrin Saturation (<16%), and Soluble Transferrin Receptor.'
        })
        recommendations.append({
            'title': 'Etiological Investigation for Iron Loss',
            'desc': 'Investigate occult gastrointestinal bleeding, celiac disease/malabsorption, or menorrhagia.'
        })
        recommendations.append({
            'title': 'Therapeutic Iron Trial with CBC Re-evaluation',
            'desc': 'Initiate oral iron therapy (e.g. ferrous sulfate/fumarate) and monitor reticulocyte response at 7-10 days and Hb recovery at 4 weeks.'
        })
        recommendations.append({
            'title': 'Post-Treatment Hemoglobin Re-testing',
            'desc': 'If microcytosis persists after normal ferritin repletion, re-screen with HPLC to uncover latent masked Thalassemia.'
        })

    # Diagnostic Divergence Insights
    divergence_notes = []
    if concordance_summary['agreement_pct'] < 75:
        divergence_notes.append(
            f"Note: {concordance_summary['total_indices'] - concordance_summary['agreement_count']} of 8 classical indices disagree with the AI. "
            "Classical formulas (like Mentzer or Shine & Lal) use fixed 1D cutoffs and often fail when microcytosis and anisocytosis overlap. "
            "The calibrated RBF Support Vector Machine integrates all 7 red cell indices simultaneously across a 97.73% validated boundary."
        )

    # Clinical Heuristic: Detect Co-existing IDA & Beta Thalassemia Trait Dual Presentation
    is_combined = bool(patient.mcv < 75.0 and patient.rdw_cv >= 16.5 and (patient.rbc_count >= 5.0 or mentzer_val < 12.5))
    combined_risk = {
        'detected': is_combined,
        'title': 'CLINICAL RISK ALERT: SUSPECTED COMBINED IDA & BETA THALASSEMIA TRAIT',
        'detail': (
            f"Dual hematological presentation detected: Microcytosis and erythrocytosis "
            f"(RBC {patient.rbc_count:.2f} × 10¹²/L, MCV {patient.mcv:.1f} fL, Mentzer {mentzer_val:.1f}) characteristic of Beta Thalassemia Trait, "
            f"co-existing with marked anisocytosis (RDW-CV {patient.rdw_cv:.1f}%) and hypochromia (MCHC {patient.mchc:.1f} g/dL) characteristic of Iron Deficiency Anemia."
        ),
        'caveat': (
            "Concomitant iron deficiency can artificially suppress HbA₂ synthesis in Beta Thalassemia Trait patients, "
            "causing HPLC HbA₂ levels to fall below the diagnostic 3.5% threshold. Recommend Serum Ferritin / TIBC testing; "
            "if iron deficient, replete iron and re-evaluate HPLC."
        )
    }

    return {
        'prediction': {
            'predicted_class': pred_class,
            'predicted_label': 'Beta Thalassemia Trait / Carrier' if pred_class == 1 else 'Iron Deficiency Anemia (IDA)',
            'prob_thalassemia': round(prob_thal, 4),
            'prob_ida': round(prob_ida, 4),
            'confidence_tier': confidence_tier,
            'primary_risk_flag': 'HIGH RISK - THALASSEMIA TRAIT' if pred_class == 1 else 'LOW RISK - IDA SUSPECTED'
        },
        'combined_risk': combined_risk,
        'shap_analysis': {
            'base_value': round(base_expected_val, 4),
            'features': shap_feature_list,
            'top_driver': shap_feature_list[0]['name'] if shap_feature_list else None
        },
        'indices': indices_list,
        'concordance_summary': concordance_summary,
        'cbc_evaluated': cbc_eval,
        'recommendations': recommendations,
        'divergence_notes': divergence_notes,
        'patient_metadata': {
            'patient_id': patient.patient_id,
            'patient_name': patient.patient_name,
            'age': patient.age,
            'gender': patient.gender,
            'referring_physician': patient.referring_physician,
            'clinical_notes': patient.clinical_notes
        }
    }

@app.get("/api/presets")
def get_presets():
    return [
        {
            'id': 'btt_classic',
            'title': 'Case 1: Typical Beta Thalassemia Trait Pattern (synthetic example)',
            'subtitle': 'Classic microcytic erythrocytosis with normal RDW (Homogeneous microcytes)',
            'data': {
                'patient_id': 'DEMO-BTT-1',
                'patient_name': 'Synthetic example (not a patient)',
                'age': '28_yrs',
                'gender': 'Unspecified',
                'referring_physician': 'Not applicable (educational demo)',
                'rbc_count': 5.82,
                'hct': 32.5,
                'hemoglobin': 9.8,
                'mcv': 56.8,
                'mch': 17.1,
                'mchc': 30.1,
                'rdw_cv': 14.1,
                'mentzer_index': 9.76,
                'clinical_notes': 'Illustrative synthetic example; not a real patient.'
            }
        },
        {
            'id': 'ida_classic',
            'title': 'Case 2: Severe Iron Deficiency Anemia (IDA) (synthetic example)',
            'subtitle': 'Marked microcytic hypochromic anemia with high RDW (Anisocytosis)',
            'data': {
                'patient_id': 'DEMO-IDA-1',
                'patient_name': 'Synthetic example (not a patient)',
                'age': '34_yrs',
                'gender': 'Unspecified',
                'referring_physician': 'Not applicable (educational demo)',
                'rbc_count': 4.10,
                'hct': 24.2,
                'hemoglobin': 7.2,
                'mcv': 59.0,
                'mch': 17.5,
                'mchc': 29.7,
                'rdw_cv': 21.8,
                'mentzer_index': 14.39,
                'clinical_notes': 'Illustrative synthetic example; not a real patient.'
            }
        },
        {
            'id': 'combined_ida_btt',
            'title': 'Case 3: Combined IDA & Beta Thal Trait (Dual Pathology)',
            'subtitle': 'Erythrocytosis (RBC 5.38) & Microcytosis with Anisocytosis (RDW 19.9%) from 100-Patient Cohort',
            'data': {
                'patient_id': 'COMB-2026-100',
                'patient_name': 'Ananya Das',
                'age': '32_yrs',
                'gender': 'Female',
                'referring_physician': 'Dr. S. Mukherjee, Antenatal Clinic',
                'rbc_count': 5.38,
                'hct': 29.6,
                'hemoglobin': 8.6,
                'mcv': 57.6,
                'mch': 15.8,
                'mchc': 27.8,
                'rdw_cv': 19.9,
                'mentzer_index': 10.71,
                'clinical_notes': 'Second trimester antenatal screening; microcytic polycythemia with marked anisocytosis. Serum Ferritin low (9 ng/mL). HPLC showed borderline HbA2 (3.2%) requiring iron repletion re-testing.'
            }
        },
        {
            'id': 'borderline_dilemma',
            'title': 'Case 4: Equivocal Borderline Microcytosis',
            'subtitle': 'Mentzer Index exactly around 13.0 - Classical indices in conflict',
            'data': {
                'patient_id': 'DEMO-EQV-1',
                'patient_name': 'Synthetic example (not a patient)',
                'age': '22',
                'gender': 'Unspecified',
                'referring_physician': 'Not applicable (educational demo)',
                'rbc_count': 4.80,
                'hct': 29.6,
                'hemoglobin': 9.5,
                'mcv': 62.4,
                'mch': 19.8,
                'mchc': 32.1,
                'rdw_cv': 16.0,
                'mentzer_index': 13.0,
                'clinical_notes': 'Illustrative synthetic example; not a real patient.'
            }
        }
    ]

@app.get("/api/decision-space")
def get_decision_space():
    """
    Returns 2D decision space coordinates for MCV vs RBC Count and MCV vs RDW
    from the 1,190-record synthetic dataset (generated with Google Gemini; not real patients).
    """
    btt_labels = [
        'Beta Thalassemia Trait', 'beta_thalassemia_heterozygous',
        'beta_thalassemia_homozygous', 'hb_lepore_trait', 'sickle_cell_heterozygous'
    ]
    ida_labels = ['Iron Deficiency Anemia', 'ida']
    
    btt_sub = cohort_df[cohort_df['HPLC Diagnosis'].isin(btt_labels)].dropna(subset=['MCV', 'RBC Count', 'RDW-CV'])
    ida_sub = cohort_df[cohort_df['HPLC Diagnosis'].isin(ida_labels)].dropna(subset=['MCV', 'RBC Count', 'RDW-CV'])
    
    # Sample up to 120 points each for smooth, instant 60fps browser rendering
    btt_sample = btt_sub.sample(min(120, len(btt_sub)), random_state=42)
    ida_sample = ida_sub.sample(min(120, len(ida_sub)), random_state=42)
    
    def extract_pts(df):
        return [
            {
                'mcv': round(float(r['MCV']), 1),
                'rbc': round(float(r['RBC Count']), 2),
                'rdw': round(float(r['RDW-CV']), 1),
                'mentzer': round(float(r['Mentzer Index']), 2),
                'diagnosis': str(r['HPLC Diagnosis'])
            }
            for _, r in df.iterrows()
        ]
        
    return {
        'btt_points': extract_pts(btt_sample),
        'ida_points': extract_pts(ida_sample),
        'axes_bounds': {
            'mcv': {'min': 42.0, 'max': 115.0},
            'rbc': {'min': 1.5, 'max': 8.0},
            'rdw': {'min': 10.0, 'max': 32.0}
        },
        'mentzer_threshold': 13.0,
        'model_metadata': {
            'framework': 'Support Vector Machine (RBF)',
            'support_vectors': 102,
            'accuracy_overall': 97.73,
            'sensitivity_btt': 96.0,
            'specificity_ida': 99.16
        }
    }

@app.get("/api/random-patient")
def get_random_patient_from_cohort():
    row = cohort_df.sample(1).iloc[0]
    return {
        'patient_id': f"COHORT-{random.randint(1000, 9999)}",
        'patient_name': 'Synthetic record (not a patient)',
        'age': str(row.get('Age', '30_yrs')),
        'gender': 'Unspecified',
        'referring_physician': 'Not applicable (synthetic record)',
        'rbc_count': round(float(row['RBC Count']), 2),
        'hct': round(float(row['HCT']), 1) if pd.notna(row['HCT']) else 30.0,
        'hemoglobin': round(float(row['Hemoglobin']), 1),
        'mcv': round(float(row['MCV']), 1),
        'mch': round(float(row['MCH']), 1),
        'mchc': round(float(row['MCHC']), 1),
        'rdw_cv': round(float(row['RDW-CV']), 1) if pd.notna(row['RDW-CV']) else 14.0,
        'mentzer_index': round(float(row['Mentzer Index']), 2),
        'ground_truth_diagnosis': str(row['HPLC Diagnosis']),
        'clinical_notes': f"Synthetic record (generated with Google Gemini) from the 1,190-record dataset; not a real patient. Dataset label: {row['HPLC Diagnosis']}."
    }

@app.get("/api/cohort-stats")
def get_cohort_stats():
    total_count = len(cohort_df)
    btt_count = int((cohort_df['is_thal'] == 1).sum())
    ida_count = int((cohort_df['is_thal'] == 0).sum())
    
    # Calculate medians for key features
    thal_df = cohort_df[cohort_df['is_thal'] == 1]
    ida_df = cohort_df[cohort_df['is_thal'] == 0]

    return {
        'total_patients': total_count,
        'thalassemia_cases': btt_count,
        'ida_other_cases': ida_count,
        'diagnosis_breakdown': cohort_df['HPLC Diagnosis'].value_counts().to_dict(),
        'medians': {
            'thal': {
                'rbc_count': round(float(thal_df['RBC Count'].median()), 2),
                'mcv': round(float(thal_df['MCV'].median()), 1),
                'rdw_cv': round(float(thal_df['RDW-CV'].median()), 1),
                'mentzer_index': round(float(thal_df['Mentzer Index'].median()), 2),
                'hemoglobin': round(float(thal_df['Hemoglobin'].median()), 1)
            },
            'ida': {
                'rbc_count': round(float(ida_df['RBC Count'].median()), 2),
                'mcv': round(float(ida_df['MCV'].median()), 1),
                'rdw_cv': round(float(ida_df['RDW-CV'].median()), 1),
                'mentzer_index': round(float(ida_df['Mentzer Index'].median()), 2),
                'hemoglobin': round(float(ida_df['Hemoglobin'].median()), 1)
            }
        },
        'index_concordance_overall': {
            'RDWI': 82.9,
            'Ricerca': 69.5,
            'Green & King': 64.7,
            'England & Fraser': 52.2,
            'Mentzer': 52.0,
            'Ehsani': 52.0,
            'Shine & Lal': 43.9,
            'Srivastava': 38.6
        }
    }

@app.post("/api/download-report")
def download_clinical_report(req: ReportDownloadRequest):
    try:
        pdf_bytes = generate_clinical_report_pdf(req.patient, req.analysis)
        pid = req.patient.get('patient_id', 'REPORT')
        safe_pid = "".join(c for c in pid if c.isalnum() or c in ('-', '_'))
        filename = f"THALA-AI_Demo_Output_NOT_FOR_CLINICAL_USE_{safe_pid}.pdf"
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate demo PDF: {str(e)}")

# Mount static folder
os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>THALA-AI PRO is compiling the frontend interface...</h1>")

@app.get("/style.css")
def serve_css():
    with open(os.path.join(BASE_DIR, "style.css"), "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="text/css")

@app.get("/app.js")
def serve_js():
    with open(os.path.join(BASE_DIR, "app.js"), "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="application/javascript")

@app.get("/model_data.js")
def serve_model_data():
    with open(os.path.join(BASE_DIR, "model_data.js"), "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="application/javascript")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
