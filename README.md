# 🌿 Tulsi Leaf Health Analyzer

An AI-powered Streamlit application for **phytopathological analysis of Tulsi (Ocimum sanctum) leaves**. The system performs leaf segmentation, calculates quantitative health parameters **exclusively on leaf tissue**, classifies the leaf as **Healthy** or **Unhealthy**, and generates detailed clinical insights using either a rule-based engine or **Groq LLaMA 3.3 70B**.

---

## ✨ Features

### 🍃 Intelligent Leaf Segmentation
- Multi-band HSV thresholding
- Morphological noise removal
- Largest connected component extraction
- Background-free leaf isolation

### 📊 Quantitative Health Analysis
Computes **8 biometric health parameters** only on segmented leaf pixels:

- Green Dominance
- Yellow/Brown Ratio
- Green Hue Coverage
- Green Channel Ratio
- HSV Saturation
- Mean Intensity
- Texture Standard Deviation
- Edge Density

### 🩺 Health Classification Engine
- Weighted scoring model
- Hard penalty gates for critical conditions
- Healthy / Unhealthy prediction
- Confidence score generation
- Severity assessment

### 🤖 AI-Generated Clinical Insights
Supports:

- Rule-based expert system (offline)
- Groq LLaMA 3.3 70B (online)

Generates:

- Clinical Summary
- Pathology Assessment
- Medical Relevance
- Phytochemical Notes
- Safety Flag
- Pharmacopoeial Compliance
- Treatment Recommendations
- Environmental Factors

### 📈 Advanced Visual Analytics
- Segmentation Overlay
- Disease Spot Mapping
- Green Dominance Visualization
- RGB Channel Decomposition
- Edge Detection Analysis
- Histograms
- Radar Charts
- Classification Breakdown Charts

### 📄 Professional PDF Reports
Generate comprehensive A4 reports including:

- Health parameters
- Classification results
- Visual charts
- Clinical observations
- Recommendations
- Safety grading

### 📦 Data Export
- JSON export support
- Easy integration with research and laboratory pipelines

---

## 🏗️ Project Structure

```text
tulsi_analyzer/
│
├── app.py                         # Streamlit entry point
├── config.py                      # Constants, thresholds, weights
│
├── core/
│   ├── __init__.py
│   ├── segmentation.py            # segment_leaf()
│   ├── analysis.py                # analyze_leaf()
│   ├── classification.py          # classify_leaf()
│   └── insights.py                # Rule-based + Groq insights
│
├── visualization/
│   ├── __init__.py
│   └── charts.py                  # Visualization utilities
│
├── report/
│   ├── __init__.py
│   └── pdf_generator.py           # PDF report generation
│
├── ui/
│   ├── __init__.py
│   ├── styles.py                  # Custom CSS styling
│   ├── image_input.py             # Upload / Camera / URL input
│   └── results.py                 # Results dashboard
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/tulsi-leaf-health-analyzer.git
cd tulsi-leaf-health-analyzer
```

### 2️⃣ Create a Virtual Environment

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🔬 Workflow

1. Upload a Tulsi leaf image
   - Local file upload
   - Camera capture
   - Image URL

2. Click **Analyze Leaf**

3. The system will:
   - Segment the leaf
   - Extract health parameters
   - Classify leaf health
   - Generate clinical insights
   - Produce visual analytics

4. Review results across multiple tabs:
   - Parameters
   - Insights
   - Processing Pipeline
   - Visual Analytics
   - Charts
   - Report

5. Download:
   - PDF Report
   - JSON Analysis Data

---

## 🧠 AI Insights Configuration

The application includes an offline rule-based diagnostic engine by default.

To enable Groq-powered insights:

1. Create an account at:
   https://console.groq.com

2. Generate an API key.

3. Update `config.py`:

```python
GROQ_API_KEY = "your_api_key_here"
```

### Model Used

```text
llama-3.3-70b-versatile
```

If the API is unavailable, the application automatically falls back to the offline expert system.

---

## 🧪 Classification Methodology

### Weighted Parameter Scoring

| Parameter | Weight |
|------------|---------|
| Green Dominance | 4 |
| Yellow/Brown Ratio | 4 |
| Green Hue Coverage | 3 |
| Green Channel Ratio | 2 |
| HSV Saturation | 2 |
| Mean Intensity | 1 |
| Texture StdDev | 1 |
| Edge Density | 1 |

### Classification Rule

```text
Healthy  → Score ≥ 55%
Unhealthy → Score < 55%
```

Critical abnormalities trigger hard penalty gates that reduce the final score regardless of overall performance.

---

## 📊 Generated Visualizations

The analysis dashboard includes:

- Leaf Segmentation Overlay
- RGB Channel Analysis
- Green Dominance Heatmap
- Disease Spot Detection
- Edge Density Visualization
- Histograms
- Radar Charts
- Classification Breakdown
- Health Metric Comparison

---

## 📄 PDF Report Contents

The generated report includes:

- Application Header
- Analysis Date & Sample ID
- Segmented Leaf Visualization
- Original Leaf Image
- Radar Chart
- Parameter Evaluation Table
- Pass/Fail Indicators
- Classification Breakdown
- Disease Spot Analysis
- Edge Analysis
- RGB Histogram
- Clinical Summary
- Pathological Indicators
- Medical Relevance
- Phytochemical Notes
- Environmental Factors
- Recommendations
- Pharmacopoeial Compliance
- Safety Flag
- Final Quality Grade

---

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|----------|
| Python 3.8+ | Core Development |
| Streamlit | Web Interface |
| OpenCV | Image Processing |
| NumPy | Numerical Computation |
| Matplotlib | Data Visualization |
| Pillow | Image Handling |
| ReportLab | PDF Generation |
| Requests | API Communication |
| Groq API | AI Insights |

---

## 📦 Requirements

```txt
streamlit>=1.25.0
numpy>=1.24.0
opencv-python-headless>=4.8.0
Pillow>=10.0.0
matplotlib>=3.7.0
reportlab>=4.0.0
requests>=2.31.0
```

---


<div align="center">

### 🌿 AI-Assisted Tulsi Health Assessment & Phytopathological Analysis

Built with ❤️ 

</div>