# 💊 Medicine Side Effect Classifier

An AI-powered web app that predicts possible side effects of medicines based on patient details — built with Python, Scikit-learn, and Streamlit.

🌐 **Live App:** [Click here to try it](your-streamlit-url)

---

## 🔴 Problem Statement
Patients often don't know what side effects a medicine may cause before taking it. This can lead to unexpected reactions and health risks.

## ✅ How It Solves It
This app predicts likely side effects based on:
- Medicine name
- Patient age & gender
- Dosage level
- Duration of usage

---

## ✨ Features
- 🔮 Predict side effects instantly
- 📊 99.8% Model Accuracy
- 🔥 Medicine vs Side Effect Heatmap
- 👥 Age Group vs Side Effect Analysis
- 🎨 Beautiful pastel UI (pink, purple, sky blue, gold)
- 📥 Top probabilities table with confidence score

---

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core language |
| Scikit-learn | Random Forest ML model |
| Pandas & NumPy | Data processing |
| Streamlit | Web UI & deployment |
| Matplotlib & Seaborn | Data visualizations |

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Sailaja-Kalle/medicine-side-effect.git
cd medicine-side-effect
```

### 2. Create virtual environment
```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate dataset
```bash
python generate_data.py
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📸 How It Works
1. Select medicine, age, gender, dosage, duration
2. Click **Predict Side Effect**
3. App shows predicted side effect with confidence %
4. View charts and heatmap for deeper insights

---

## 👩‍💻 Built By 
[LinkedIn](https://linkedin.com/in/sailaja-kalle) | [GitHub](https://github.com/Sailaja-Kalle)

---

⭐ If you find this useful, give it a star!
