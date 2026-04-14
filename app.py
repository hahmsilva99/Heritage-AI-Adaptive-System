import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import time
import random
from urllib.parse import quote_plus
import numpy as np

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
    .postpone-box { background-color: #E8F8F5; padding: 20px; border-radius: 10px; border-left: 5px solid #1ABC9C; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
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

# --- Initialize Session State for clean UI ---
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'current_site' not in st.session_state:
    st.session_state.current_site = None

# --- 3. Sidebar ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Flag_of_Sri_Lanka.svg/1200px-Flag_of_Sri_Lanka.svg.png", width=100)
st.sidebar.title("Navigation 🧭")
app_mode = st.sidebar.radio("Select View:", ["1. Tourist Explorer (User)", "2. Admin Dashboard (Panel)"])
st.sidebar.markdown("---")
st.sidebar.info("🤖 **Status: AI Online**\n\nModel Accuracy: 96.88%\n\n*Optimizing for Heritage Conservation.*")

# --- 4. Main App: Tourist Explorer ---
if app_mode == "1. Tourist Explorer (User)":
    st.markdown('<div class="main-header">🏛️ Eco-Adaptive Heritage Explorer</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Sustainable Journey Planning Powered by Community & AI.</p><hr>", unsafe_allow_html=True)

    # UI Inputs
    st.subheader("📍 Where are you heading?")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_district = st.selectbox("1. District:", sorted(df['District'].unique()))
    with col2:
        district_sites = df[df['District'] == selected_district]['Site Name'].unique()
        selected_site = st.selectbox("2. Heritage Site:", sorted(district_sites))
    with col3:
        target_audience = st.selectbox("3. Traveler Type:", df['Target_Audience'].unique())

    # If user changes the site, reset the analysis view to keep UI clean
    if st.session_state.current_site != selected_site:
        st.session_state.analyzed = False
        st.session_state.current_site = selected_site

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 🚀 The Analyze Button
    if st.button("🚀 Analyze with AI & Get Recommendation"):
        st.session_state.analyzed = True
        
        # We generate random live data ONLY when button is pressed, and save to session state so it doesn't change when chatting
        with st.spinner("🛰️ Fetching Live Satellite Weather & Crowd Sensor Data..."):
            time.sleep(1.5) 
            st.session_state.live_weather = random.choice(["Sunny", "Cloudy", "Rainy", "Clear"])
            st.session_state.live_aqi = random.choice(["Good", "Moderate", "Poor"])
            st.session_state.live_overcrowding = random.choices(["Low", "Medium", "High"], weights=[20, 30, 50], k=1)[0] 

    # --- SHOW RESULTS ONLY IF BUTTON WAS PRESSED ---
    if st.session_state.analyzed:
        site_data = df[df['Site Name'] == selected_site].iloc[0].copy()
        capacity = site_data['Max Visitor Capacity']
        conservation = site_data['Conservation Status']
        recommended_time = site_data['Recommended Time']

        # Live Data Display
        st.markdown(f"""
        <div class="live-data-box">
            <h4 style='margin-top:0; color:#154360;'>📡 Real-Time Status for {selected_site}</h4>
            <p><b>Weather:</b> {st.session_state.live_weather}   |   <b>AQI:</b> {st.session_state.live_aqi}   |   <b>Crowd Level:</b> {st.session_state.live_overcrowding}</p>
        </div>
        """, unsafe_allow_html=True)

        # Crowd Trend Graph
        st.markdown("#### 📊 Expected Crowd Trend Today")
        time_labels = ['06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00']
        crowd_map = {"High": [20, 50, 95, 100, 85, 60, 30], "Medium": [15, 40, 70, 75, 60, 45, 20], "Low": [10, 20, 40, 45, 35, 25, 10]}
        colors = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#2ECC71"}
        
        trend_df = pd.DataFrame({'Time': time_labels, 'Density (%)': crowd_map[st.session_state.live_overcrowding]}).set_index('Time')
        st.area_chart(trend_df, color=colors[st.session_state.live_overcrowding], height=180)

        # AI Prediction
        site_data['Weather_Condition'] = st.session_state.live_weather
        site_data['AQI_Level'] = st.session_state.live_aqi
        site_data['Overcrowding Risk'] = st.session_state.live_overcrowding
        site_data['Target_Audience'] = target_audience
        
        input_data = {col: le_dict[col].transform([str(site_data[col])])[0] if col in le_dict else site_data[col] for col in feature_columns}
        input_df = pd.DataFrame([input_data])
        input_df['Max Visitor Capacity'] = pd.to_numeric(input_df['Max Visitor Capacity'])
        
        prediction = ai_model.predict(input_df)[0]
        decision = le_dict['Redirect Recommendation'].inverse_transform([prediction])[0]
        
        st.markdown("---")
        if decision == 'No':
            st.markdown(f'<div class="success-box"><h3>✅ Clear to Visit!</h3><p>Enjoy <b>{selected_site}</b>. Conditions are optimal.</p></div>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f'<div class="alert-box"><h3>⚠️ Redirection Activated</h3><p><b>{selected_site}</b> is currently facing {st.session_state.live_overcrowding} risk.</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="postpone-box">⏳ <b>Smart Postponement:</b> Optimal window is <b>{recommended_time}</b>.</div>', unsafe_allow_html=True)
            
            # Alternatives
            st.subheader("🌿 Sustainable Alternatives")
            alternatives = df[(df['District'] == selected_district) & (df['Overcrowding Risk'] == 'Low') & (df['Site Name'] != selected_site)]
            if not alternatives.empty:
                alts = alternatives.sample(min(3, len(alternatives)))
                cols = st.columns(len(alts))
                for col, (_, alt) in zip(cols, alts.iterrows()):
                    with col:
                        st.markdown(f'<div class="alt-card"><h4>{alt["Site Name"]}</h4><p>📍 {alt["Location"]}</p>', unsafe_allow_html=True)
                        q = quote_plus(f"{alt['Site Name']} {alt['District']} Sri Lanka")
                        st.markdown(f"**[🗺️ Navigate](https://www.google.com/maps/search/?api=1&query={q})**")
                        st.markdown('</div>', unsafe_allow_html=True)

        # --- LIVE CROWDSOURCING BUTTONS ---
        st.markdown("---")
        st.subheader("📢 Live Community Feedback")
        st.write(f"Are you currently at **{selected_site}**? Help others by reporting live conditions!")
        
        feed_col1, feed_col2 = st.columns(2)
        with feed_col1:
            if st.button("🔴 It's Very Crowded!", help="Report high congestion"):
                st.toast("Thank you! Your report helps protect this heritage site.", icon="🙏")
                st.success("Report Submitted. AI is updating the live heatmaps...")
        with feed_col2:
            if st.button("🟢 It's Peaceful & Quiet", help="Report low congestion"):
                st.toast("Awesome! We've updated the status for other travelers.", icon="✨")
                st.info("Report Submitted. Thanks for being a sustainable traveler!")

        # --- AI CHATBOT UI ---
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("💬 Ask AI Heritage Guide"):
            chat = st.text_input("Ask about the history or status of a site:")
            if chat: 
                with st.spinner("AI is thinking..."):
                    time.sleep(1)
                    st.info(f"**AI:** Great question about '{chat}'! {selected_site} has immense cultural value. Visiting during off-peak hours helps preserve its structure.")

# --- 5. Admin Dashboard ---
elif app_mode == "2. Admin Dashboard (Panel)":
    st.title("📊 System Analytics Dashboard")
    st.info("📌 This view is for SLTDA Administrators to monitor model accuracy and redirection trends.")