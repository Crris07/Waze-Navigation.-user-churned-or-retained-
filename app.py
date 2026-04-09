import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ── Load artifacts ──
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

with open('features.json') as f:
    config = json.load(f)

FEATURES = config['features']
DEVICE_COL = config['device_col']
THRESHOLD = config['threshold']


def safe_divide(num, denom):
    result = num / denom
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    result.fillna(0, inplace=True)
    return result


def engineer_features(df):
    """Apply same feature engineering as training."""
    df = df.copy()
    df['km_per_drive'] = safe_divide(df['driven_km_drives'], df['drives'])
    df['drives_per_day'] = safe_divide(df['drives'], df['activity_days'])
    df['sessions_per_day'] = safe_divide(df['sessions'], df['activity_days'])
    df['percent_days_driving'] = safe_divide(df['driving_days'], df['activity_days'])
    df['activity_to_driving_ratio'] = safe_divide(df['activity_days'], df['driving_days'])

    # Device dummies — align to training columns
    device_dummies = pd.get_dummies(df['device'], prefix='device')
    df = pd.concat([df, device_dummies], axis=1)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURES]


# Streamlit UI
st.set_page_config(page_title="Waze Churn Predictor", page_icon="🚗", layout="wide")

st.title("Waze User Churn Predictor")
st.markdown("Upload a CSV of user data to predict who's likely to churn.")

# ── Sidebar: manual single-user prediction ──
st.sidebar.header("🔍 Single User Prediction")

with st.sidebar.form("single_user"):
    sessions = st.number_input("Sessions", min_value=0, value=40)
    drives = st.number_input("Drives", min_value=0, value=25)
    total_sessions = st.number_input("Total Sessions", min_value=0.0, value=200.0)
    driven_km = st.number_input("Driven KM", min_value=0.0, value=3000.0)
    duration_min = st.number_input("Duration (minutes)", min_value=0.0, value=1800.0)
    activity_days = st.number_input("Activity Days", min_value=0, value=15)
    driving_days = st.number_input("Driving Days", min_value=0, value=12)
    n_days_onboarding = st.number_input("Days After Onboarding", min_value=0, value=1200)
    device = st.selectbox("Device", ["iPhone", "Android"])
    submitted = st.form_submit_button("Predict")

if submitted:
    user_df = pd.DataFrame([{
        'sessions': sessions,
        'drives': drives,
        'total_sessions': total_sessions,
        'driven_km_drives': driven_km,
        'duration_minutes_drives': duration_min,
        'activity_days': activity_days,
        'driving_days': driving_days,
        'n_days_after_onboarding': n_days_onboarding,
        'device': device
    }])

    X_user = engineer_features(user_df)
    X_user_scaled = pd.DataFrame(scaler.transform(X_user), columns=X_user.columns)
    prob = model.predict_proba(X_user_scaled)[0, 1]
    label = "Churned" if prob >= THRESHOLD else " Retained"

    st.sidebar.metric("Churn Probability", f"{prob:.1%}")
    st.sidebar.markdown(f"**Prediction:** {label}")
    st.sidebar.caption(f"Threshold: {THRESHOLD:.3f}")

# Main area: batch CSV upload
st.header("Batch Prediction")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df_upload = pd.read_csv(uploaded)
    st.write(f"**Loaded:** {len(df_upload)} rows")

    # Check required columns
    required = ['sessions', 'drives', 'total_sessions', 'driven_km_drives',
                'duration_minutes_drives', 'activity_days', 'driving_days',
                'n_days_after_onboarding', 'device']
    missing = [c for c in required if c not in df_upload.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        X_batch = engineer_features(df_upload)
        X_batch_scaled = pd.DataFrame(scaler.transform(X_batch), columns=X_batch.columns)
        probs = model.predict_proba(X_batch_scaled)[:, 1]

        df_upload['churn_probability'] = probs
        df_upload['prediction'] = ['Churned' if p >= THRESHOLD else 'Retained' for p in probs]

        # Summary metrics
        n_churn = (probs >= THRESHOLD).sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Users", len(df_upload))
        col2.metric("Predicted Churners", n_churn)
        col3.metric("Churn Rate", f"{n_churn / len(df_upload):.1%}")

        # Show results
        st.dataframe(
            df_upload[['ID', 'churn_probability', 'prediction']].sort_values(
                'churn_probability', ascending=False
            ) if 'ID' in df_upload.columns else
            df_upload[['churn_probability', 'prediction']].sort_values(
                'churn_probability', ascending=False
            ),
            use_container_width=True
        )

        # Download button
        csv_out = df_upload.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇Download Predictions CSV",
            csv_out,
            "churn_predictions.csv",
            "text/csv"
        )
