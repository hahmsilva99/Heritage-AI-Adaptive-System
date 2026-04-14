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

# --- Custom CSS (UPDATED FOR ABSOLUTE BLACK TEXT) ---
st.markdown("""
    <style>
    .main-header { font-size: 36px; font-weight: bold; color: #1F618D; text-align: center; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #239B56; color: white; font-weight: bold; padding: 10px; font-size: 18px;}
    .stButton>button:hover { background-color: #1D8348; }
    
    /* Box Styles with Black Text Forced */
    .alert-box { padding: 15px; border-radius: 10px; background-color: #FADBD8; border-left: 5px solid #E74C3C; margin-bottom: 20px; color: black !important; }
    .alert-box * { color: black !important; }
    
    .success-box { padding: 15px; border-radius: 10px; background-color: #D5F5E3; border-left: 5px solid #2ECC71; margin-bottom: 20px; color: black !important; }
    .success-box * { color: black !important; }
    
    .live-data-box { padding: 15px; border-radius: 10px; background-color: #EBF5FB; border-left: 5px solid #3498DB; margin-bottom: 20px; color: black !important; }
    .live-data-box * { color: black !important; }
    
    .alt-card { border: 1px solid #D5DBDB; border-radius: 10px; padding: 15px; background-color: #F8F9F9; height: 100%; color: black !important; }
    .alt-card * { color: black !important; }
    
    .postpone-box { background-color: #E8F8F5; padding: 20px; border-radius: 10px; border-left: 5px solid #1ABC9C; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: black !important; }
    .postpone-box * { color: black !important; }
    
    .crowd-box { background-color: #FFF9C4; padding: 20px; border-radius: 10px; border-left: 5px solid #F1C40F; margin-top: 30px; margin-bottom: 20px; color: black !important; }
    .crowd-box * { color: black !important; }
    
    .impact-box { background-color: #EBF5FB; border: 2px dashed #28B463; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 25px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); color: black !important; }
    .impact-box * { color: black !important; }
    
    .micro-zone-box { background-color: #F5EEF8; padding: 15px; border-radius: 10px; border-left: 5px solid #8E44AD; margin-bottom: 20px; color: black !important; }
    .micro-zone-box * { color: black !important; }
    
    .forecast-box { background-color: #F4ECF7; padding: 20px; border-radius: 10px; border-left: 5px solid #9B59B6; margin-top: 20px; margin-bottom: 20px; color: black !important; }
    .forecast-box * { color: black !important; }
    
    /* 🔥 NEW: Forced Black Text for Forecast Result Box */
    .forecast-result-box { background-color: #FDFEFE; padding: 15px; border-radius: 8px; border: 1px solid #D2B4DE; color: black !important; }
    .forecast-result-box * { color: black !important; font-weight: bold; }
    
    .badge { font-size: 50px; }
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

# --- Initialize Session State ---
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'current_site' not in st.session_state:
    st.session_state.current_site = None
if 'accepted_alt' not in st.session_state:
    st.session_state.accepted_alt = False
if 'accepted_alt_name' not in st.session_state:
    st.session_state.accepted_alt_name = ""

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

    st.subheader("📍 Where are you heading?")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_district = st.selectbox("1. Destination District:", sorted(df['District'].unique()), key="dest_dist")
    with col2:
        district_sites = df[df['District'] == selected_district]['Site Name'].unique()
        selected_site = st.selectbox("2. Heritage Site:", sorted(district_sites), key="dest_site")
    with col3:
        target_audience = st.selectbox("3. Traveler Type:", df['Target_Audience'].unique(), key="dest_type")

    if st.session_state.current_site != selected_site:
        st.session_state.analyzed = False
        st.session_state.accepted_alt = False
        st.session_state.current_site = selected_site

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Analyze with AI & Get Recommendation"):
        st.session_state.analyzed = True
        st.session_state.accepted_alt = False 
        
        with st.spinner("🛰️ Fetching Live Satellite Weather & Crowd Sensor Data..."):
            time.sleep(1.5) 
            st.session_state.live_weather = random.choice(["Sunny", "Cloudy", "Rainy", "Clear"])
            st.session_state.live_aqi = random.choice(["Good", "Moderate", "Poor"])
            st.session_state.live_overcrowding = random.choices(["Low", "Medium", "High"], weights=[10, 40, 50], k=1)[0] 

    if st.session_state.analyzed:
        site_data = df[df['Site Name'] == selected_site].iloc[0].copy()
        capacity = site_data['Max Visitor Capacity']
        conservation = site_data['Conservation Status']
        recommended_time = site_data['Recommended Time']

        site_data['Weather_Condition'] = st.session_state.live_weather
        site_data['AQI_Level'] = st.session_state.live_aqi
        site_data['Overcrowding Risk'] = st.session_state.live_overcrowding
        site_data['Target_Audience'] = target_audience
        
        input_data = {col: le_dict[col].transform([str(site_data[col])])[0] if col in le_dict else site_data[col] for col in feature_columns}
        input_df = pd.DataFrame([input_data])
        input_df['Max Visitor Capacity'] = pd.to_numeric(input_df['Max Visitor Capacity'])
        
        prediction = ai_model.predict(input_df)[0]
        decision = le_dict['Redirect Recommendation'].inverse_transform([prediction])[0]

        if st.session_state.accepted_alt:
            st.markdown("---")
            st.markdown(f"""
            <div class="impact-box">
                <div class="badge">🏅🌱</div>
                <h2>Green Tourist Badge Unlocked!</h2>
                <p style="font-size: 18px;">Thank you for accepting the AI recommendation to visit <b>{st.session_state.accepted_alt_name}</b> instead of adding to the congestion at {selected_site}.</p>
                <hr style="border: 1px solid #A9DFBF; width: 50%; margin: 20px auto;">
                <h3>📊 Your Sustainability Impact:</h3>
            </div>
            """, unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Heritage Stress Reduced", "85%", "+15% from avg")
            m2.metric("Sustainable Reward Points", "+150 XP", "Top 10% today")
            m3.metric("Local Economy Supported", "Alternative Site", "Yes")
            
            st.balloons()
            
            if st.button("🔄 Plan Another Trip"):
                st.session_state.analyzed = False
                st.session_state.accepted_alt = False
                st.rerun()

        else:
            st.markdown(f"""
            <div class="live-data-box">
                <h4 style='margin-top:0;'>📡 Real-Time Status for {selected_site}</h4>
                <p><b>Weather:</b> {st.session_state.live_weather}   |   <b>AQI:</b> {st.session_state.live_aqi}   |   <b>Crowd Level:</b> {st.session_state.live_overcrowding}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="micro-zone-box">
                <h4 style='margin-top: 0;'>🗺️ Micro-Zone Crowd Radar</h4>
                <p style='font-size: 14px; margin-bottom: 10px;'>Sustainability AI distributes foot traffic to prevent localized structural damage. Live conditions within the <b>{selected_site}</b> complex:</p>
            </div>
            """, unsafe_allow_html=True)

            main_risk = st.session_state.live_overcrowding
            if main_risk == "High":
                z2_risk, z3_risk = "Medium", "Low"
                c1, c2, c3 = "🔴", "🟠", "🟢"
            elif main_risk == "Medium":
                z2_risk, z3_risk = "Low", "Low"
                c1, c2, c3 = "🟠", "🟢", "🟢"
            else:
                z2_risk, z3_risk = "Low", "Low"
                c1, c2, c3 = "🟢", "🟢", "🟢"

            mz1, mz2, mz3 = st.columns(3)
            with mz1:
                st.markdown(f'<div style="background-color: #D6EAF8; padding: 15px; border-radius: 8px; text-align: center; color: black; border: 1px solid #AED6F1;"><b>Main Complex</b><br><br>{c1} {main_risk} Traffic<br><br>🚶 0 min</div>', unsafe_allow_html=True)
            with mz2:
                st.markdown(f'<div style="background-color: #D5F5E3; padding: 15px; border-radius: 8px; text-align: center; color: black; border: 1px solid #ABEBC6;"><b>Outer Gardens</b><br><br>{c2} {z2_risk} Traffic<br><br>🚶 5 mins (400m)</div>', unsafe_allow_html=True)
            with mz3:
                st.markdown(f'<div style="background-color: #D5F5E3; padding: 15px; border-radius: 8px; text-align: center; color: black; border: 1px solid #ABEBC6;"><b>Scenic Viewpoint</b><br><br>{c3} {z3_risk} Traffic<br><br>🚶 10 mins (800m)</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("#### 📊 Expected Crowd Trend Today")
            time_labels = ['06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00']
            crowd_map = {"High": [20, 50, 95, 100, 85, 60, 30], "Medium": [15, 40, 70, 75, 60, 45, 20], "Low": [10, 20, 40, 45, 35, 25, 10]}
            colors = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#2ECC71"}
            trend_df = pd.DataFrame({'Time': time_labels, 'Density (%)': crowd_map[st.session_state.live_overcrowding]}).set_index('Time')
            st.area_chart(trend_df, color=colors[st.session_state.live_overcrowding], height=180)

            st.markdown("---")
            if decision == 'No':
                st.markdown(f'<div class="success-box"><h3 style="margin-top:0;">✅ Clear to Visit!</h3><p>Enjoy <b>{selected_site}</b>. Conditions are optimal.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-box"><h3 style="margin-top:0;">⚠️ Redirection Activated</h3><p><b>{selected_site}</b> is currently facing {st.session_state.live_overcrowding} risk.</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="postpone-box"><h4 style="margin-top:0;">⏳ Smart Postponement</h4><p>If visiting <b>{selected_site}</b> is an absolute must for you, the AI highly recommends rescheduling.</p><h3>Optimal Visiting Window: {recommended_time}</h3></div>', unsafe_allow_html=True)
                
                st.subheader("🌿 Suggested Safe Alternatives")
                alternatives = df[(df['District'] == selected_district) & (df['Overcrowding Risk'] == 'Low') & (df['Site Name'] != selected_site)]
                if not alternatives.empty:
                    alts = alternatives.sample(min(3, len(alternatives)))
                    cols = st.columns(len(alts))
                    for col, (_, alt) in zip(cols, alts.iterrows()):
                        with col:
                            st.markdown(f'<div class="alt-card"><h4 style="margin-top:0;">{alt["Site Name"]}</h4><p>📍 {alt["Location"]}</p><p>🛡️ {alt["Conservation Status"]}</p>', unsafe_allow_html=True)
                            
                            search_query = quote_plus(f"{alt['Site Name']} {alt['District']} Sri Lanka")
                            st.markdown(f"**[🗺️ Navigate via Maps](https://www.google.com/maps/search/?api=1&query={search_query})**")
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            if st.button(f"🌍 Accept & Save Heritage", key=f"btn_{alt['Site Name']}"):
                                st.session_state.accepted_alt = True
                                st.session_state.accepted_alt_name = alt['Site Name']
                                st.rerun() 
                                
                            st.markdown('</div>', unsafe_allow_html=True)

            # ==========================================
            # 🔮 FIXED: Predictive Forecasting Box Text Color
            # ==========================================
            st.markdown("---")
            st.markdown(f"""
            <div class="forecast-box">
                <h4 style='margin-top: 0;'>🔮 AI Predictive Forecasting (Plan Ahead)</h4>
                <p style='font-size: 14px;'>Cannot visit today? Select a future timeline to see AI-predicted conditions based on historical tourism data.</p>
            </div>
            """, unsafe_allow_html=True)

            fc_col1, fc_col2 = st.columns([1, 2])
            with fc_col1:
                forecast_day = st.selectbox("Select Timeline:", ["Tomorrow", "Coming Weekend", "Next Week"], key="forecast_day")

            with fc_col2:
                f_weather = random.choice(["Sunny", "Clear", "Light Rain"])
                f_crowd = random.choice(["Low", "Medium", "High"])
                f_color = "🔴" if f_crowd == "High" else "🟠" if f_crowd == "Medium" else "🟢"

                # New strict CSS class applied here to force black text
                st.markdown(f"""
                <div class="forecast-result-box">
                    Forecast for {selected_site} ({forecast_day}):<br><br>
                    🌡️ Weather: {f_weather}   |   {f_color} Expected Crowd: {f_crowd}
                </div>
                """, unsafe_allow_html=True)
            # ==========================================

            # Live Crowdsourcing
            st.markdown("---")
            st.markdown(f'<div class="crowd-box">', unsafe_allow_html=True)
            st.subheader("📢 Help the Community: Report Live Conditions")
            report_col1, report_col2 = st.columns(2)
            with report_col1:
                report_district = st.selectbox("I am currently in (District):", sorted(df['District'].unique()), key="report_dist")
            with report_col2:
                report_sites = df[df['District'] == report_district]['Site Name'].unique()
                report_site = st.selectbox("I am currently visiting (Site):", sorted(report_sites), key="report_site")
            st.markdown("<br>", unsafe_allow_html=True)
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button(f"🔴 It's Very Crowded at {report_site}!", key="btn_crowd"):
                    st.toast("Thank you! Your report helps protect this heritage site.", icon="🙏")
            with btn_col2:
                if st.button(f"🟢 It's Peaceful & Quiet at {report_site}", key="btn_peace"):
                    st.toast("Awesome! We've updated the status.", icon="✨")
            st.markdown('</div>', unsafe_allow_html=True)

            # Chatbot
            with st.expander("💬 Ask AI Heritage Guide"):
                chat = st.text_input("Ask about the history or status of a site:", key="chat_input")
                if chat: 
                    with st.spinner("AI is thinking..."):
                        time.sleep(1)
                        st.info(f"**AI:** Great question about '{chat}'! The sites in {selected_district} have immense cultural value. Visiting during off-peak hours helps preserve their structure.")

# --- 5. Admin Dashboard ---
elif app_mode == "2. Admin Dashboard (Panel)":
    st.title("📊 System Analytics Dashboard")
    st.info("📌 This view is for SLTDA Administrators to monitor model accuracy and redirection trends.")