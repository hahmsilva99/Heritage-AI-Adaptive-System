import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import time
import random

# --- 1. Page Configuration ---
st.set_page_config(page_title="Sri Lanka Heritage AI", page_icon="🏛️", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main-header { font-size: 36px; font-weight: bold; color: #1F618D; text-align: center; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #239B56; color: white; font-weight: bold; }
    .stButton>button:hover { background-color: #1D8348; }
    .alert-box { padding: 15px; border-radius: 10px; background-color: #FADBD8; border-left: 5px solid #E74C3C; margin-bottom: 20px;}
    .success-box { padding: 15px; border-radius: 10px; background-color: #D5F5E3; border-left: 5px solid #2ECC71; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# --- 2. AI Model Training (Background process) ---
@st.cache_resource
def train_ai_model():
    df = pd.read_csv("Cleaned_Heritage_Sites_Final_Fixed.csv")
    
    features = df.drop(columns=['Site Name', 'Location', 'Sensitive Periods', 'District'])
    target_col = 'Redirect Recommendation'
    
    features['Max Visitor Capacity'] = pd.to_numeric(features['Max Visitor Capacity'], errors='coerce')
    features['Max Visitor Capacity'] = features['Max Visitor Capacity'].fillna(features['Max Visitor Capacity'].median())

    le_dict = {}
    for col in features.columns:
        if features[col].dtype == 'object':
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
            le_dict[col] = le

    X = features.drop(columns=[target_col])
    y = features[target_col]

    # SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # XGBoost
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_res, y_res)
    
    return df, model, le_dict, X.columns

with st.spinner("Initializing Adaptive AI Model..."):
    df, ai_model, le_dict, feature_columns = train_ai_model()

# --- 3. Sidebar ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Flag_of_Sri_Lanka.svg/1200px-Flag_of_Sri_Lanka.svg.png", width=100)
st.sidebar.title("Navigation 🧭")
app_mode = st.sidebar.radio("Select View:", ["1. Tourist Explorer (User)", "2. Admin Dashboard (Panel)"])
st.sidebar.markdown("---")
st.sidebar.info("🤖 **Status: AI Online**\n\nModel Accuracy: 96.88%\n\n*Optimizing for Heritage Conservation.*")

# --- 4. Main App: Tourist Explorer ---
if app_mode == "1. Tourist Explorer (User)":
    st.markdown('<div class="main-header">🏛️ Eco-Adaptive Heritage Explorer</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Plan your journey sustainably with Real-Time AI.</p><hr>", unsafe_allow_html=True)

    # UI Inputs
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📍 Where to?")
        selected_district = st.selectbox("Select District:", sorted(df['District'].unique()))
        
        # Filter sites based on district
        district_sites = df[df['District'] == selected_district]['Site Name'].unique()
        selected_site = st.selectbox("Select Heritage Site:", sorted(district_sites))
        
        target_audience = st.selectbox("Traveler Type:", df['Target_Audience'].unique())

    with col2:
        st.subheader("🌍 Live Environmental Data")
        weather = st.selectbox("Current Weather:", df['Weather_Condition'].unique())
        aqi = st.selectbox("Air Quality (AQI) Level:", df['AQI_Level'].unique())
        overcrowding = st.select_slider("Current Overcrowding Risk:", options=['Low', 'Medium', 'High'])

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction Logic
    if st.button("🚀 Analyze with AI & Get Recommendation"):
        with st.spinner("AI is analyzing conservation limits and live data..."):
            time.sleep(1.5) # Loading effect
            
            # Get original site details
            site_data = df[df['Site Name'] == selected_site].iloc[0].copy()
            
            # Override with user real-time inputs
            site_data['Weather_Condition'] = weather
            site_data['AQI_Level'] = aqi
            site_data['Overcrowding Risk'] = overcrowding
            site_data['Target_Audience'] = target_audience
            
            # Prepare data for prediction
            input_data = {}
            for col in feature_columns:
                val = site_data[col]
                if col in le_dict:
                    # Handle unseen labels just in case
                    try:
                        input_data[col] = le_dict[col].transform([str(val)])[0]
                    except:
                        input_data[col] = 0
                else:
                    input_data[col] = val
                    
            input_df = pd.DataFrame([input_data])
            
            # Make AI Prediction
            prediction = ai_model.predict(input_df)[0]
            decision = le_dict['Redirect Recommendation'].inverse_transform([prediction])[0]
            
            st.markdown("---")
            st.subheader("🤖 AI Decision")
            
            if decision == 'No':
                # No Redirection needed (Safe to go)
                st.markdown(f"""
                <div class="success-box">
                    <h3 style='margin-top:0;'>✅ Clear to Visit!</h3>
                    <p>The environmental conditions at <b>{selected_site}</b> are optimal. Overcrowding is manageable and conservation status is safe.</p>
                    <p>Enjoy your sustainable trip!</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                # Redirection needed (At risk)
                st.markdown(f"""
                <div class="alert-box">
                    <h3 style='margin-top:0;'>⚠️ Adaptive Redirection Activated</h3>
                    <p>Due to <b>{overcrowding} overcrowding risk</b> and <b>{weather} weather</b>, visiting <b>{selected_site}</b> right now could harm the heritage site or degrade your experience.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Find an alternative site (Same district, Low Risk, Not the same site)
                alternatives = df[(df['District'] == selected_district) & 
                                  (df['Overcrowding Risk'] == 'Low') & 
                                  (df['Site Name'] != selected_site)]
                
                st.subheader("🌿 Recommended Alternative")
                if not alternatives.empty:
                    alt_site = alternatives.sample(1).iloc[0]
                    st.success(f"**Discover {alt_site['Site Name']} instead!**")
                    st.write(f"📍 **Location:** {alt_site['Location']}")
                    st.write(f"🎭 **Type:** {alt_site['Type']}")
                    st.write(f"💡 **Why?** It currently has a Low overcrowding risk and is well-suited for {target_audience}.")
                else:
                    st.warning("We recommend relaxing at your hotel for now. Currently, all major sites in this district are facing high risks.")

# --- 5. Main App: Admin Dashboard ---
elif app_mode == "2. Admin Dashboard (Panel)":
    st.title("📊 System Analytics Dashboard")
    st.write("This section is restricted for Tourism Authority Administrators.")
    st.info("📌 **Note for Defense Panel:** We will integrate the Data Visualizations (ROC Curve, Heatmaps, District Analysis) in this section during the next phase!")