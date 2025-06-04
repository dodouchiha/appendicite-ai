import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Carica il modello Random Forest
@st.cache_resource
def load_model():
    return joblib.load("random_forest_appendicite.pkl")

model = load_model()

# Titolo app
st.title("Calcolo del Rischio di Appendicite Acuta")
st.subheader("Supporto alla decisione clinica con AI per donne in età fertile")

# Inserimento parametri clinico-laboratoristici
st.markdown("### Inserisci i parametri clinici")

età = st.number_input("Età", min_value=15, max_value=50, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
tc = st.number_input("Temperatura Corporea (°C)", min_value=35.0, max_value=42.0, step=0.1)
wbc = st.number_input("Globuli Bianchi (x10³/µL)", min_value=2.0, max_value=30.0, step=0.1)
nlr = st.number_input("NLR", min_value=0.1, max_value=20.0, step=0.1)
tlr = st.number_input("TLR", min_value=1.0, max_value=1000.0, step=1.0)
pcr = st.number_input("PCR", min_value=0.0, max_value=500.0, step=0.1)
durata = st.number_input("Durata sintomi (h)", min_value=0, max_value=72, step=1)
alvarado = st.slider("Alvarado Score", 0, 10)
air = st.slider("AIR Score", 0, 12)

# Calcolo predizione
if st.button("Calcola rischio"):
    input_data = pd.DataFrame({
        "eta": [età],
        "bmi": [bmi],
        "tc": [tc],
        "wbc": [wbc],
        "nlr": [nlr],
        "tlr": [tlr],
        "pcr": [pcr],
        "dur_sint": [durata],
        "alvarado": [alvarado],
        "air": [air]
    })
    
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.markdown(f"### Rischio stimato: **{prob*100:.1f}%**")
    st.write("Diagnosi probabile:", "Appendicite" if pred == 1 else "Non Appendicite")

    if prob > 0.8:
        st.error("Alta probabilità. Considerare appendicectomia.")
    elif prob > 0.5:
        st.warning("Rischio intermedio. Ulteriori accertamenti raccomandati.")
    else:
        st.success("Bassa probabilità. Monitoraggio clinico.")