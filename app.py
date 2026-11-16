import joblib, numpy as np, pandas as pd, streamlit as st

@st.cache_resource
def load():
    tfidf, model = joblib.load("baseline.joblib")
    return tfidf, model, np.array(tfidf.get_feature_names_out())

tfidf, model, words = load()

st.title("Complaint Triage")
st.caption("Predicts which financial product a complaint is about (TF-IDF + Logistic Regression).")
text = st.text_area("Paste a customer complaint", height=180)

if st.button("Classify") and text.strip():
    x = tfidf.transform([text])
    probs = model.predict_proba(x)[0]
    top = np.argsort(probs)[::-1][:3]
    st.table(pd.DataFrame({"product": model.classes_[top], "score": probs[top].round(2)}))

    contrib = x.toarray()[0] * model.coef_[top[0]]
    best = np.argsort(contrib)[::-1][:8]
    best = best[contrib[best] > 0]
    st.write("Words pushing toward the top prediction:", ", ".join(words[best]))
    st.caption("Scores rank products; they are not calibrated probabilities (class weights inflate small classes).")
