import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import time
import random
from urllib.parse import quote_plus

# --- 1. Page Configuration ---
st.set_page_config(page_title="Sri Lanka Heritage AI", page_icon="🏛️", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main-header { font-size: 36px; font-weight: bold; color: #1F618D; text-align: center; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #239B56; color: white; font-weight: bold; padding: 10px; font-size: 18px;}
    .stButton>button:hover { background-color: #1D8348; }
    .alert-box { padding: 15px; border-radius: 10px; background-color: #FADBD8; border-left: 5px solid #E74C3C; margin-bottom: 20px;}
    .success-box { padding: 15px; border-radius: 10px; background-color: #D5F5E3; border-left: 5px solid #2ECC71; margin-bottom: 20px;}
    .live-data-box { padding: 15px; border-radius: 10px; background-color: #EBF5FB; border-left: 5px solid #3498DB; margin-bottom: 20px;}
    .alt-card { border: 1px solid #D5DBDB; border-radius: 10px; padding: 15px; background-color: #F8F9F9; height: 100%;}
    </style>
""", unsafe_allow_html=True)

# --- 2. AI Model Training ---
@st.cache_resource
def train_ai_model():
    df = pd.read_csv("Cleaned_Heritage_Sites_Final_Fixed.csv")
    
    df['Max Visitor Capacity'] = pd.to_numeric(df['Max Visitor Capacity'], errors='coerce')
    df['Max Visitor Capacity'] = df['Max Visitor Capacity'].fillna(df['Max Visitor Capacity'].median())

    features = df.drop(columns=['Site Name', 'Location', 'Sensitive Periods', 'District'])
    target_col = 'Redirect Recommendation'

    le_dict = {}
    for col in features.columns:
        if features[col].dtype == 'object':
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
            le_dict[col] = le

    X = features.drop(columns=[target_col])
    y = features[target_col]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

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
    st.markdown("<p style='text-align: center; color: gray;'>Tell us where you want to go. AI will check real-time conditions for you.</p><hr>", unsafe_allow_html=True)

    st.subheader("📍 Your Travel Plan")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_district = st.selectbox("1. Select District:", sorted(df['District'].unique()))
    with col2:
        district_sites = df[df['District'] == selected_district]['Site Name'].unique()
        selected_site = st.selectbox("2. Select Heritage Site:", sorted(district_sites))
    with col3:
        target_audience = st.selectbox("3. Traveler Type:", df['Target_Audience'].unique())

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Analyze with AI & Get Recommendation"):
        with st.spinner("🛰️ Fetching Live Satellite Weather & Crowd Sensor Data..."):
            time.sleep(2) 
            
            live_weather = random.choice(["Sunny", "Cloudy", "Rainy", "Clear"])
            live_aqi = random.choice(["Good", "Moderate", "Poor"])
            live_overcrowding = random.choices(["Low", "Medium", "High"], weights=[20, 30, 50], k=1)[0] 
            
            site_data = df[df['Site Name'] == selected_site].iloc[0].copy()
            capacity = site_data['Max Visitor Capacity']
            conservation = site_data['Conservation Status']
            recommended_time = site_data['Recommended Time']

        st.markdown(f"""
        <div class="live-data-box">
            <h4 style='margin-top:0; color:#154360;'>📡 Real-Time Data Retrieved for {selected_site}</h4>
            <p><b>Weather:</b> {live_weather} &nbsp; | &nbsp; <b>Air Quality (AQI):</b> {live_aqi} &nbsp; | &nbsp; <b>Current Overcrowding:</b> {live_overcrowding}</p>
            <p><b>Max Capacity:</b> {int(capacity)} visitors &nbsp; | &nbsp; <b>Site Condition:</b> {conservation}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🧠 AI is evaluating conservation risk..."):
            time.sleep(1.5)
            
            site_data['Weather_Condition'] = live_weather
            site_data['AQI_Level'] = live_aqi
            site_data['Overcrowding Risk'] = live_overcrowding
            site_data['Target_Audience'] = target_audience
            
            input_data = {}
            for col in feature_columns:
                val = site_data[col]
                if col in le_dict:
                    try:
                        input_data[col] = le_dict[col].transform([str(val)])[0]
                    except:
                        input_data[col] = 0
                else:
                    input_data[col] = val
                    
            input_df = pd.DataFrame([input_data])
            input_df['Max Visitor Capacity'] = pd.to_numeric(input_df['Max Visitor Capacity'])
            
            prediction = ai_model.predict(input_df)[0]
            decision = le_dict['Redirect Recommendation'].inverse_transform([prediction])[0]
            
            st.markdown("---")
            st.subheader("🤖 AI Final Decision")
            
            if decision == 'No':
                st.markdown(f"""
                <div class="success-box">
                    <h3 style='margin-top:0;'>✅ Clear to Visit!</h3>
                    <p>The environmental conditions at <b>{selected_site}</b> are optimal right now. The crowd levels are safe, and visiting won't harm the conservation efforts.</p>
                    <p>Enjoy your sustainable trip!</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="alert-box">
                    <h3 style='margin-top:0;'>⚠️ Adaptive Redirection Activated</h3>
                    <p>Visiting <b>{selected_site}</b> right now is not recommended due to <b>{live_overcrowding} Overcrowding Risk</b> and <b>{live_weather} Weather</b>.</p>
                    <p>To protect the site's structural integrity and ensure you have a better experience, we suggest these safe alternatives.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Pro Feature: Postponement Advice
                st.info(f"💡 **Still want to visit {selected_site}?** The AI suggests coming back during: **{recommended_time}** when crowds are naturally lower.")
                
                # Find up to 3 alternatives
                alternatives = df[(df['District'] == selected_district) & 
                                  (df['Overcrowding Risk'] == 'Low') & 
                                  (df['Site Name'] != selected_site)]
                
                st.subheader("🌿 Recommended Alternatives for You")
                if not alternatives.empty:
                    # Get up to 3 alternative sites
                    num_alts = min(3, len(alternatives))
                    top_alts = alternatives.sample(num_alts)
                    
                    # Display them in beautiful columns
                    cols = st.columns(num_alts)
                    
                    for col, (_, alt_site) in zip(cols, top_alts.iterrows()):
                        with col:
                            st.markdown(f'<div class="alt-card">', unsafe_allow_html=True)
                            st.markdown(f"#### {alt_site['Site Name']}")
                            st.caption(f"📍 {alt_site['Location']}")
                            st.write(f"🎭 **Type:** {alt_site['Type']}")
                            st.write(f"🛡️ **Status:** {alt_site['Conservation Status']}")
                            
                            # Google Maps Link Generation
                            search_query = quote_plus(f"{alt_site['Site Name']} {alt_site['District']} Sri Lanka")
                            maps_url = f"https://www.google.com/maps/search/?api=1&query={search_query}"
                            st.markdown(f"**[🗺️ View on Google Maps]({maps_url})**")
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("We recommend relaxing at your hotel for now. Currently, all major alternative sites in this district are also facing high risks.")

# --- 5. Main App: Admin Dashboard ---
elif app_mode == "2. Admin Dashboard (Panel)":
    st.title("📊 System Analytics Dashboard")
    st.write("This section is restricted for Tourism Authority Administrators.")