import streamlit as st
import numpy as np
import joblib

# Load models and scaler
model1 = joblib.load('model1_linear_reg.pkl')
model2_reg = joblib.load('model2_voting_multioutput.pkl')
model2_cls = joblib.load('model2_gnb_cls.pkl')
scaler = joblib.load('scaler_model2.pkl')
label_encoder = joblib.load('shape_label_encoder.pkl')

st.title("Particle Shape and Property Prediction App")

# --- User input ---
min_val = st.number_input('Enter Min value', min_value=0.0, step=0.0001, format="%.4f")
mid_val = st.number_input('Enter Mid value', min_value=0.0, step=0.0001, format="%.4f")
max_val = st.number_input('Enter Max value', min_value=0.0, step=0.0001, format="%.4f")

if st.button("Predict"):
    # === Step 1: Model 1 Prediction ===
    input1 = np.array([[max_val, mid_val, min_val]])
    predicted_volumes = model1.predict(input1)[0]
    av, chv, sa = predicted_volumes
    st.success(f"Predicted Volumes: Actual Volume={av:.3f}, Convex Hull Volume={chv:.3f}, Surface Area={sa:.3f}")

    # === Step 2: Calculate 5 Features ===
    EI = mid_val / max_val if max_val != 0 else 0
    FI = min_val / mid_val if mid_val != 0 else 0
    AR = (EI + FI) / 2
    CI = av / chv if chv != 0 else 0
    S = S = ((36 * np.pi * (av ** 2)) ** (1/3)) / sa if sa != 0 else 0

    feature_array = np.array([[EI, FI, AR, CI, S]])
    feature_scaled = scaler.transform(feature_array)

    # === Step 3: Model 2 Regression ===
    reg_preds = model2_reg.predict(feature_scaled)[0]
    friction_angle, void_ratio = reg_preds

    # === Step 4: Model 2 Classification ===
    shape_pred_encoded = model2_cls.predict(feature_scaled)[0]
    shape_pred = label_encoder.inverse_transform([shape_pred_encoded])[0]

    # === Step 5: Final Output ===
    st.subheader("🔎 Final Predictions")
    st.write(f"**Friction Angle:** {friction_angle:.2f}")
    st.write(f"**Void Ratio:** {void_ratio:.3f}")
    st.write(f"**Shape Class:** {shape_pred}")
