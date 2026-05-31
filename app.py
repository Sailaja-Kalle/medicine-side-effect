import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Medicine Side Effect Classifier", page_icon="💊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Nunito', sans-serif !important; }
.stApp {
    background: linear-gradient(135deg, #fce4f0 0%, #e8f4fd 30%, #ede8fa 60%, #fdf6e3 100%) !important;
    background-attachment: fixed !important;
}
.block-container {
    background: rgba(255,255,255,0.75); backdrop-filter: blur(12px);
    border-radius: 20px; padding: 2rem 2.5rem !important;
    box-shadow: 0 8px 40px rgba(130,100,180,0.12);
    border: 1px solid rgba(255,255,255,0.9);
}
h1 { background: linear-gradient(90deg, #c56cd6, #5b8dee, #f9a825);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     font-size: 2.2rem !important; font-weight: 800 !important; }
h2, h3 { color: #5a4a8a !important; font-weight: 700 !important; }
.stButton > button {
    background: linear-gradient(135deg, #f9a825, #f06292) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: 15px !important; padding: 10px 24px !important;
    box-shadow: 0 4px 18px rgba(249,168,37,0.35) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #f06292, #7c4dff) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(124,77,255,0.4) !important;
}
.stSelectbox > div > div {
    border-radius: 10px !important; border: 1.5px solid #c4b5fd !important;
    background: rgba(255,255,255,0.9) !important;
}
.stSlider > div > div > div { background: linear-gradient(90deg, #87ceeb, #c56cd6) !important; }
.stMetric { background: linear-gradient(135deg, #fce4f0, #ede8fa) !important;
            border-radius: 12px !important; padding: 12px !important;
            border: 1.5px solid #e0d7ff !important; }
.stSuccess { border-radius: 10px !important; border-left: 4px solid #4ade80 !important; }
.stWarning { border-radius: 10px !important; border-left: 4px solid #f9a825 !important; }
hr { background: linear-gradient(90deg, #87ceeb, #ffb6c1, #d8b4fe, #fde68a) !important;
     height: 2px !important; border: none !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: linear-gradient(#c4b5fd, #fbcfe8); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

st.title("💊 Medicine Side Effect Classifier")
st.markdown("Predict possible side effects based on medicine, dosage, age and more.")
st.divider()

@st.cache_data
def load_and_train():
    df = pd.read_csv("medicine_data.csv")
    le_gender  = LabelEncoder()
    le_med     = LabelEncoder()
    le_dosage  = LabelEncoder()
    le_effect  = LabelEncoder()
    df["gender_enc"]  = le_gender.fit_transform(df["gender"])
    df["medicine_enc"]= le_med.fit_transform(df["medicine"])
    df["dosage_enc"]  = le_dosage.fit_transform(df["dosage"])
    df["effect_enc"]  = le_effect.fit_transform(df["side_effect"])
    X = df[["age","gender_enc","medicine_enc","dosage_enc","duration_days"]]
    y = df["effect_enc"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le_gender, le_med, le_dosage, le_effect, df, acc

model, le_gender, le_med, le_dosage, le_effect, df, acc = load_and_train()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("🔍 Enter Patient Details")
    age      = st.slider("Age", 18, 80, 30)
    gender   = st.selectbox("Gender", ["Male", "Female"])
    medicine = st.selectbox("Medicine", sorted(df["medicine"].unique()))
    dosage   = st.selectbox("Dosage Level", ["Low", "Medium", "High"])
    duration = st.slider("Duration (days)", 1, 30, 7)

    if st.button("🔮 Predict Side Effect", use_container_width=True, type="primary"):
        input_data = pd.DataFrame({
            "age":          [age],
            "gender_enc":   [le_gender.transform([gender])[0]],
            "medicine_enc": [le_med.transform([medicine])[0]],
            "dosage_enc":   [le_dosage.transform([dosage])[0]],
            "duration_days":[duration]
        })
        pred     = model.predict(input_data)[0]
        proba    = model.predict_proba(input_data)[0]
        result   = le_effect.inverse_transform([pred])[0]
        confidence = round(max(proba) * 100, 1)

        st.session_state["result"] = result
        st.session_state["confidence"] = confidence
        st.session_state["proba"] = proba

if "result" in st.session_state:
    st.divider()
    result = st.session_state["result"]
    confidence = st.session_state["confidence"]
    proba = st.session_state["proba"]
    r1, r2, r3 = st.columns([2, 1, 1])
    with r1:
        if result == "No Side Effect":
            st.success(f"✅ Predicted: **{result}**")
        else:
            st.warning(f"⚠️ Predicted Side Effect: **{result}**")
    with r2:
        st.metric("Confidence", f"{confidence}%")
    with r3:
        st.metric("Model Accuracy", f"{round(acc*100,1)}%")
    proba_df = pd.DataFrame({
        "Side Effect": le_effect.classes_,
        "Probability": (proba * 100).round(1)
    }).sort_values("Probability", ascending=False).head(5)
    st.markdown("**Top Probabilities:**")
    st.dataframe(proba_df, use_container_width=True, hide_index=True)
    st.divider()

with col2:
    st.subheader("📊 Dataset Insights")
    st.metric("Model Accuracy", f"{round(acc*100,1)}%")
    st.metric("Total Records", len(df))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df["side_effect"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Side Effect Distribution")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=45)

    df["medicine"].value_counts().plot(kind="bar", ax=axes[1], color="coral")
    axes[1].set_title("Medicine Distribution")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='x', rotation=45)

    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha='right', fontsize=8)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha='right', fontsize=8)
    plt.tight_layout(pad=2.0)
st.pyplot(fig)

st.markdown("### 🔥 Medicine vs Side Effect Heatmap")
pivot = df.groupby(["medicine","side_effect"]).size().unstack(fill_value=0)
fig2, ax2 = plt.subplots(figsize=(12, 5))
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax2, linewidths=0.5)
ax2.set_title("Medicine vs Side Effect", fontsize=13)
ax2.set_xlabel("")
ax2.set_ylabel("")
plt.xticks(rotation=30, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
st.pyplot(fig2)

st.markdown("### 👥 Age Group vs Side Effect")
df["age_group"] = pd.cut(df["age"], bins=[18,30,45,60,80], labels=["18-30","31-45","46-60","61-80"])
pivot2 = df.groupby(["age_group","side_effect"]).size().unstack(fill_value=0)
fig3, ax3 = plt.subplots(figsize=(12, 4))
pivot2.plot(kind="bar", ax=ax3, colormap="Set2")
ax3.set_title("Age Group vs Side Effect", fontsize=13)
ax3.set_xlabel("Age Group")
ax3.legend(loc="upper right", fontsize=7)
plt.xticks(rotation=0)
plt.tight_layout()
st.pyplot(fig3)