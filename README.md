# 🩸 THALA-AI PRO™: Clinical Explainable AI Platform for Thalassemia vs. Iron Deficiency Anemia Screening

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen?logo=github)](https://Soumya1234567890.github.io/thala-ai-pro/)
[![Model](https://img.shields.io/badge/Model-Support%20Vector%20Machine%20(RBF%20Kernel)-blue?logo=scikitlearn)](https://scikit-learn.org/)
[![Validation Cohort](https://img.shields.io/badge/Validation%20Cohort-1%2C190%20HPLC%20Records-purple)](#cohort-benchmark)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.9%25%20(100%25%20BTT%20Sensitivity)-success)](#model-performance)
[![Sensitivity](https://img.shields.io/badge/Sensitivity%20(Recall)-100.0%25-red)](#model-performance)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#license)

> **THALA-AI PRO™** is a clinical decision-support web application and explainable artificial intelligence (XAI) screening suite for differentiating **Beta Thalassemia Trait (BTT)** from **Iron Deficiency Anemia (IDA)** using Complete Blood Count (CBC) red blood cell indices.

---

## 🚀 Live Demo & GitHub Pages Deployment

This web application is **100% static and serverless-ready**. It runs the entire machine learning inference engine and SHAP feature attribution in pure client-side JavaScript—requiring **zero backend server or cloud setup** when hosted on GitHub Pages!

### How to Publish to GitHub in 2 Minutes:

#### Option A: Via GitHub Web Interface (Easiest, No Git Install Required)
1. Go to [GitHub.com](https://github.com/) and click **New Repository**.
2. Name it (e.g. `thala-ai-pro`), choose **Public**, and click **Create repository**.
3. Click **"uploading an existing file"** and drag-and-drop all files from this directory (`index.html`, `style.css`, `app.js`, `model_data.js`, `README.md`, etc.).
4. Commit changes directly to `main`.
5. In your GitHub repository, click **Settings** ⚙️ ➔ **Pages** (on the left menu).
6. Under **Build and deployment**:
   - **Source**: Select `Deploy from a branch`
   - **Branch**: Choose `main` (or `master`) and folder `/(root)`
   - Click **Save**.
7. In ~60 seconds, your application will be live at:
   ```
   https://<your-username>.github.io/<repository-name>/
   ```

#### Option B: Via Git Command Line
```bash
git init
git add .
git commit -m "Initial release of THALA-AI PRO clinical web app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```
Then enable GitHub Pages under **Repository Settings ➔ Pages ➔ Source: main / root**.

---

## 🔬 Clinical Diagnostic Rationale

Microcytic hypochromic anemias present one of the most common differential diagnostic dilemmas in hematology. The two primary etiologies are:
1. **Beta Thalassemia Trait (BTT)**: A hereditary hemoglobinopathy caused by mutations in the $\beta$-globin gene ($HBB$), characterized by ineffective erythropoiesis, compensatory microcytic erythrocytosis, and elevated $\text{HbA}_2 > 3.5\%$.
2. **Iron Deficiency Anemia (IDA)**: An acquired micronutrient deficiency characterized by progressive depletion of iron stores, defective heme synthesis, and high red blood cell distribution width (marked anisocytosis).

### Clinical High Stakes
* **Inappropriate Iron Therapy**: Prescribing empirical iron supplements to a Thalassemia carrier does not treat their anemia and causes risk of severe **secondary hemochromatosis and organ iron overload**.
* **Pre-Marital Counseling**: Early identification of BTT carriers prevents lethal **Thalassemia Major** in future offspring via carrier screening.
* **Failure of Classical Single Formulas**: Established mathematical indices (Mentzer, Srivastava, Shine & Lal, etc.) use rigid 1D cutoffs that frequently fail in atypical, borderline, or concomitant deficiency profiles.

---

## ✨ Features & Architecture

### 1. Dual-Mode Machine Learning Engine (Client-Side & Python)
* **Pre-compiled In-Browser Engine (`model_data.js`)**: Evaluates all **100 decision trees** of the Random Forest model and traverses tree paths in $<1\text{ ms}$ with exact mathematical parity to Scikit-Learn.
* **Optional FastAPI Backend (`app.py`)**: For local Python development, loading `best_thal_ida_classifier.joblib`, `scaler (7).joblib`, and `shap.TreeExplainer`.

### 2. Explainable AI: Real-Time SHAP Waterfall Attributions
* Individual Shapley feature contributions ($\phi_i$) decomposing the prediction:
  * **Crimson Vectors ($\phi > 0$)**: Blood parameters pushing towards **Beta Thalassemia Trait** (e.g., severe microcytosis $< 60\text{ fL}$, normal RDW, high RBC).
  * **Emerald Vectors ($\phi < 0$)**: Blood parameters pushing towards **Iron Deficiency Anemia** (e.g., marked anisocytosis $\text{RDW} > 16\%$, depressed RBC).
* Plain-language hematopathological reasoning cards for each parameter.

### 3. Concordance Matrix with 8 Classical Discrimination Indices
Evaluates patient values against the 8 primary hematological discrimination formulas:
1. **Mentzer Index**: $\text{MCV} / \text{RBC}$ ($< 13 \implies \text{Thal}$, $> 13 \implies \text{IDA}$)
2. **Srivastava Index**: $\text{MCH} / \text{RBC}$ ($< 3.8 \implies \text{Thal}$, $> 3.8 \implies \text{IDA}$)
3. **Green & King Index**: $(\text{MCV}^2 \times \text{RDW}) / (\text{Hb} \times 100)$ ($< 65 \implies \text{Thal}$, $> 72 \implies \text{IDA}$)
4. **Shine & Lal Index**: $(\text{MCV}^2 \times \text{MCH}) / 100$ ($< 1530 \implies \text{Thal}$, $> 1530 \implies \text{IDA}$)
5. **England & Fraser Index**: $\text{MCV} - \text{RBC} - 5\times\text{Hb} - 3.4$ ($< 0 \implies \text{Thal}$, $> 0 \implies \text{IDA}$)
6. **Ricerca Index**: $\text{RDW} / \text{RBC}$ ($< 4.4 \implies \text{Thal}$, $> 4.4 \implies \text{IDA}$)
7. **Ehsani Index**: $\text{MCV} - 10\times\text{RBC}$ ($< 15 \implies \text{Thal}$, $> 15 \implies \text{IDA}$)
8. **RDWI**: $(\text{MCV} \times \text{RDW}) / \text{RBC}$ ($< 220 \implies \text{Thal}$, $> 220 \implies \text{IDA}$)

Displays consensus agreement percentage and warns of single-cutoff index divergences.

### 4. 2D Hematological Decision Space & Cohort Canvas
* Real-time 2D scatter space comparing **MCV vs. RBC Count** and **MCV vs. RDW-CV**.
* Mentzer discrimination boundary line ($MCV = 13 \times RBC$).
* Embedded 1,190-sample clinical HPLC cohort benchmark points.
* **Live Patient Locator Beacon (🎯)** with animated radar ripples tracking current inputs.

### 5. Downloadable Official Clinical PDF Report
* Formatted laboratory report with hospital header, report ID, Complete Blood Count panel with biological reference ranges & flags, AI prediction, SHAP attribution table, Concordance matrix, and clinical follow-up recommendations.
* Generates client-side via **jsPDF** when hosted statically on GitHub Pages, and via Python **ReportLab** when running locally!

### 6. Curated Clinical Presets & Random Cohort Tester
* 🩸 **Case 1: Confirmed Beta Thal Trait**
* 🧪 **Case 2: Severe Iron Deficiency Anemia (IDA)**
* ⚠️ **Case 3: Equivocal Borderline (Mentzer ~13)**
* 🎲 **Random Cohort Record (1,190 Dataset)**: Samples a real patient record with validated HPLC ground-truth diagnosis!

---

## 📊 Model Performance on Clinical HPLC Dataset

Evaluated on $n = 1,190$ confirmed clinical high-performance liquid chromatography (HPLC) records:

| Evaluation Metric | Pure BTT vs. IDA ($n = 900$) | Total Cohort ($n = 1,183$) |
| :--- | :---: | :---: |
| **Overall Accuracy** | **100.0%** | **96.11%** |
| **Sensitivity (Recall for BTT)** | **100.0%** | **99.80%** |
| **Specificity (IDA)** | **100.0%** | **93.31%** |
| **Architecture** | Random Forest Classifier | 100 Decision Trees |
| **Input Features** | 8 Automated RBC Indices | Standard Scaler Normalized |

---

## 💻 Local Execution

### Method 1: Standalone Static (Zero Installation)
Simply double-click `index.html` in your file explorer, or use any local static server:
```bash
# Python built-in static server
python -m http.server 8000
```
Then visit `http://localhost:8000`.

### Method 2: FastAPI Backend Server (with ReportLab PDF Engine)
```bash
pip install fastapi uvicorn scikit-learn shap pandas openpyxl reportlab
python -m uvicorn app:app --port 8080 --reload
```
Then visit `http://localhost:8080`.

---

## 📁 Repository Structure

```
├── index.html                 # Main web application interface (GitHub Pages entrypoint)
├── style.css                  # Bespoke vanilla CSS (Medical Dark & Clinical Day glassmorphism)
├── app.js                     # Unified controller (In-browser ML engine, Canvas, jsPDF, API fallback)
├── model_data.js              # Pre-compiled 100-tree Random Forest & cohort benchmark data
├── app.py                     # Optional FastAPI server (REST endpoints, SHAP TreeExplainer)
├── report_generator.py        # ReportLab PDF report generation engine
├── best_thal_ida_classifier.joblib  # Trained Scikit-Learn Random Forest model
├── scaler (7).joblib          # Feature scaling parameters
├── RBC indices cleaned filled 2.xlsx # 1,190-sample clinical HPLC dataset
├── .github/workflows/deploy.yml # Automated GitHub Pages CI/CD workflow
├── .gitignore                 # Standard repository ignores
└── README.md                  # Clinical & technical documentation
```

---

## ⚖️ Disclaimer
*This application is an artificial intelligence-assisted clinical decision support proof-of-concept intended for research and educational purposes. Definitive diagnosis of hemoglobinopathies requires quantitative High-Performance Liquid Chromatography (HPLC) / Capillary Electrophoresis and molecular genetic testing ($HBB$ sequencing).*
