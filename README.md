# 🩸 THALA-AI PRO™: Educational Explainable AI Prototype for Thalassemia Trait vs. Iron Deficiency Anemia

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen?logo=github)](https://Soumya1234567890.github.io/thala-ai-pro/)
[![Model](https://img.shields.io/badge/Model-Support%20Vector%20Machine%20(RBF%20Kernel)-blue?logo=scikitlearn)](https://scikit-learn.org/)
[![Data](https://img.shields.io/badge/Data-Synthetic%20(Google%20Gemini)-orange)](#model-performance)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy%20(synthetic%2C%20n%3D220)-97.7%25-success)](#model-performance)
[![Sensitivity](https://img.shields.io/badge/BTT%20Sensitivity%20(synthetic%20test)-96.0%25-red)](#model-performance)
[![Use](https://img.shields.io/badge/Use-Educational%20only%2C%20not%20for%20clinical%20use-critical)](#model-performance)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#license)

> [!WARNING]
> **Educational research prototype trained on synthetic data generated with Google Gemini. Not a medical device; not for clinical use; no laboratory accreditation.**

> **THALA-AI PRO™** is an educational research prototype and explainable artificial intelligence (XAI) demonstration for differentiating **Beta Thalassemia Trait (BTT)** from **Iron Deficiency Anemia (IDA)** using Complete Blood Count (CBC) red blood cell indices. It was trained and tested **only on synthetic records generated with Google Gemini** (no real patient data).

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
git commit -m "Initial release of THALA-AI PRO educational demo"
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
* **Pre-compiled In-Browser Engine (`model_data.js`)**: Evaluates the **Support Vector Machine (RBF kernel, 102 support vectors)** on **7 inputs** (RBC, Hb, HCT, MCV, MCH, MCHC, RDW-CV) in $<1\text{ ms}$, using the scikit-learn decision function and Platt-scaled probability $P(\text{BTT}) = 1/(1+e^{\,\text{probA}\cdot f - \text{probB}})$.
* **Optional FastAPI Backend (`app.py`)**: For local Python development, loading `best_thal_ida_classifier.joblib` and `scaler (7).joblib`; feature attributions use a marginal (one-feature-at-a-time) decomposition of the SVM decision function.

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
* Embedded points from the 1,190-record **synthetic** dataset (generated with Google Gemini; not real patients).
* **Live Patient Locator Beacon (🎯)** with animated radar ripples tracking current inputs.

### 5. Downloadable Educational Demo PDF
* Printable demo sheet (clearly marked **not a clinical report / not for clinical use**) with the entered CBC values, reference ranges & flags, model prediction, attribution table, concordance matrix, and general confirmatory work-up notes.
* Generates client-side via **jsPDF** when hosted statically on GitHub Pages, and via Python **ReportLab** when running locally!

### 6. Example Presets & Random Synthetic-Record Tester
* 🩸 **Case 1: Typical Beta Thal Trait Pattern**
* 🧪 **Case 2: Severe Iron Deficiency Anemia (IDA)**
* ⚠️ **Case 3: Equivocal Borderline (Mentzer ~13)**
* 🎲 **Random Cohort Record (1,190 Dataset)**: Samples a record from the 1,190 dataset together with its ground-truth label!

---

## 📊 Model Performance (Synthetic Held-Out Test Set)

The dataset consists of **1,190 synthetic records generated with Google Gemini** (no real patients). The deployed model was evaluated on a **held-out synthetic test set of 220 records**. These figures describe performance on synthetic data only; they are **not** clinical validation.

| Evaluation Metric | Held-out synthetic test set ($n = 220$) |
| :--- | :---: |
| **Overall Accuracy** | **97.7%** (215/220; 95% CI 94.8–99.0) |
| **Sensitivity (BTT)** | **96.0%** (97/101) |
| **Specificity (IDA / non-BTT)** | **99.2%** (118/119) |
| **ROC-AUC** | **0.994** |
| **Architecture** | Support Vector Machine (RBF kernel, scikit-learn SVC) |
| **Input Features** | 7: RBC, Hb, HCT, MCV, MCH, MCHC, RDW-CV (StandardScaler-normalised) |

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
├── model_data.js              # Pre-compiled Support Vector Machine (RBF) & cohort benchmark data
├── app.py                     # Optional FastAPI server (REST endpoints, SHAP Kernel Attribution)
├── report_generator.py        # ReportLab PDF report generation engine
├── best_thal_ida_classifier.joblib  # Trained Scikit-Learn Support Vector Machine (SVM) model
├── scaler (7).joblib          # Feature scaling parameters
├── RBC indices cleaned filled 2.xlsx # 1,190-record synthetic dataset (Google Gemini-generated)
├── .github/workflows/deploy.yml # Automated GitHub Pages CI/CD workflow
├── .gitignore                 # Standard repository ignores
└── README.md                  # Technical documentation
```

---

## ⚖️ Disclaimer
**Educational research prototype trained on synthetic data generated with Google Gemini. Not a medical device; not for clinical use; no laboratory accreditation.**

*This application is an educational proof-of-concept intended for research and teaching purposes only. It was trained and tested exclusively on synthetic data, has not been clinically validated, and must not be used to diagnose, screen, or manage any patient. Definitive diagnosis of hemoglobinopathies requires quantitative High-Performance Liquid Chromatography (HPLC) / Capillary Electrophoresis and molecular genetic testing ($HBB$ sequencing).*
