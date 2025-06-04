
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Carica il modello
@st.cache_resource
def load_model():
    return joblib.load("random_forest_appendicite.pkl")

model = load_model()

st.title("Calcolo del Rischio di Appendicite Acuta")
st.subheader("Supporto decisionale con AI per donne in età fertile")

st.markdown("### Inserisci i parametri clinici")

età = st.number_input("Età", 15, 50)
bmi = st.number_input("BMI", 10.0, 50.0)
tc = st.number_input("Temperatura Corporea (°C)", 35.0, 42.0)
wbc = st.number_input("Globuli Bianchi (x10³/µL)", 2.0, 30.0)
nlr = st.number_input("NLR", 0.1, 20.0)
tlr = st.number_input("TLR", 1.0, 1000.0)
pcr = st.number_input("PCR", 0.0, 500.0)
durata = st.number_input("Durata sintomi (h)", 0, 72)
alvarado = st.slider("Alvarado Score", 0, 10)
air = st.slider("AIR Score", 0, 12)

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
