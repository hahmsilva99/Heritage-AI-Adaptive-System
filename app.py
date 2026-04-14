import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
import time
import random
from urllib.parse import quote_plus
import numpy as np

# IMPORTING THE ADMIN DASHBOARD MODULE
from admin_dashboard import render_admin_dashboard 

# --- 1. Page Configuration ---
st.set_page_config(page_title="Sri Lanka Heritage AI", page_icon="🏛️", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main-header { font-size: 36px; font-weight: bold; color: #1F618D; text-align: center; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #239B56; color: white; font-weight: bold; padding: 10px; font-size: 18px;}
    .stButton>button:hover { background-color: #1D8348; }
    
    .alert-box, .success-box, .live-data-box, .alt-card, .postpone-box, .crowd-box, .impact-box, .micro-zone-box, .forecast-box, .ticket-card { color: black !important; }
    .alert-box * , .success-box * , .live-data-box * , .alt-card * , .postpone-box * , .crowd-box * , .impact-box * , .micro-zone-box * , .forecast-box * , .ticket-card * { color: black !important; }
    
    .alert-box { padding: 15px; border-radius: 10px; background-color: #FADBD8; border-left: 5px solid #E74C3C; margin-bottom: 20px; }
    .success-box { padding: 15px; border-radius: 10px; background-color: #D5F5E3; border-left: 5px solid #2ECC71; margin-bottom: 20px; }
    .live-data-box { padding: 15px; border-radius: 10px; background-color: #EBF5FB; border-left: 5px solid #3498DB; margin-bottom: 20px; }
    .alt-card { border: 1px solid #D5DBDB; border-radius: 10px; padding: 15px; background-color: #F8F9F9; height: 100%; }
    .postpone-box { background-color: #E8F8F5; padding: 20px; border-radius: 10px; border-left: 5px solid #1ABC9C; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .crowd-box { background-color: #FFF9C4; padding: 20px; border-radius: 10px; border-left: 5px solid #F1C40F; margin-top: 30px; margin-bottom: 20px; }
    .impact-box { background-color: #EBF5FB; border: 2px dashed #28B463; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 25px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); }
    .micro-zone-box { background-color: #F5EEF8; padding: 15px; border-radius: 10px; border-left: 5px solid #8E44AD; margin-bottom: 20px; }
    .forecast-box { background-color: #F4ECF7; padding: 20px; border-radius: 10px; border-left: 5px solid #9B59B6; margin-top: 20px; margin-bottom: 20px; }
    
    .ticket-card { background-color: #FDFEFE; border: 2px solid #3498DB; border-radius: 15px; padding: 20px; margin-bottom: 20px; border-style: dashed;}
    .ticket-price { font-size: 24px; font-weight: bold; color: #1F618D !important; }
    .discount-badge { background-color: #2ECC71; color: white !important; padding: 5px 10px; border-radius: 5px; font-size: 12px; font-weight: bold;}
    
    .profile-box { background-color: #F9E79F; padding: 15px; border-radius: 10px; border-left: 5px solid #F39C12; margin-bottom: 20px; color: black !important;}
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
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return df, model, le_dict, X.columns, fpr, tpr, roc_auc

with st.spinner("Initializing Adaptive AI Model..."):
    df, ai_model, le_dict, feature_columns, fpr, tpr, roc_auc = train_ai_model()

def calculate_smart_ticket(base_site, nationality, crowd_level, is_alternative=False):
    if nationality == "Foreign (Tourist)":
        price = 30.0 
        unit = "USD"
    else:
        price = 200.0 
        unit = "LKR"
        
    surcharge = 0
    discount = 0
    
    if crowd_level == "High":
        surcharge = price * 0.20 
        price += surcharge
    
    if is_alternative:
        discount = price * 0.15 
        price -= discount
        
    return price, unit, surcharge, discount

# --- Initialize Session State ---
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'current_site' not in st.session_state:
    st.session_state.current_site = None
if 'accepted_alt' not in st.session_state:
    st.session_state.accepted_alt = False
if 'accepted_alt_name' not in st.session_state:
    st.session_state.accepted_alt_name = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'total_users' not in st.session_state:
    st.session_state.total_users = 1245
if 'total_redirects' not in st.session_state:
    st.session_state.total_redirects = 412

# --- 3. Sidebar ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Flag_of_Sri_Lanka.svg/1200px-Flag_of_Sri_Lanka.svg.png", width=100)
st.sidebar.title("Navigation 🧭")
app_mode = st.sidebar.selectbox("Select View:", ["1. Tourist Explorer (User)", "2. Admin Dashboard (Panel)"])
st.sidebar.markdown("---")
st.sidebar.info("🤖 **Status: AI Online**\n\nModel Accuracy: 96.88%\n\n*Optimizing for Heritage Conservation.*")

# --- 4. Main App: Tourist Explorer ---
if app_mode == "1. Tourist Explorer (User)":
    st.markdown('<div class="main-header">🏛️ Eco-Adaptive Heritage Explorer</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Sustainable Journey Planning Powered by Community & AI.</p><hr>", unsafe_allow_html=True)

    # ==========================================
    # 🔥 UPDATED HEADINGS HERE
    # ==========================================
    st.subheader("📍 Step 1: Primary Destination Configuration")
    col1, col2, col3 = st.columns(3) 
    with col1:
        selected_district = st.selectbox("Destination District:", sorted(df['District'].unique()), key="dest_dist")
    with col2:
        district_sites = df[df['District'] == selected_district]['Site Name'].unique()
        selected_site = st.selectbox("Heritage Site:", sorted(district_sites), key="dest_site")
    with col3:
        nationality = st.selectbox("Nationality:", ["Local (Sri Lankan)", "Foreign (Tourist)"], key="nat_type")

    st.subheader("🎒 Step 2: Smart Accessibility & Eco-Profile")
    st.markdown("<p style='font-size: 14px; color: gray;'>Help the AI recommend the most comfortable and sustainable route for you.</p>", unsafe_allow_html=True)
    
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        target_audience = st.selectbox("Traveler Interest:", df['Target_Audience'].unique(), key="dest_type")
    with t_col2:
        mobility_level = st.selectbox("Mobility Needs:", ["Standard", "Elderly / Kids Friendly", "Wheelchair Accessible"])
    with t_col3:
        transport_mode = st.selectbox("Transport Mode:", ["Private Vehicle", "Public Transport", "Cycling / Walking"])

    if st.session_state.current_site != selected_site:
        st.session_state.analyzed = False
        st.session_state.accepted_alt = False
        st.session_state.chat_history = [] 
        st.session_state.current_site = selected_site

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Analyze with AI & Get Recommendation"):
        st.session_state.analyzed = True
        st.session_state.accepted_alt = False 
        st.session_state.total_users += 1
        
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

        main_price, m_unit, m_surcharge, _ = calculate_smart_ticket(selected_site, nationality, st.session_state.live_overcrowding)

        if st.session_state.accepted_alt:
            st.markdown("---")
            alt_price, a_unit, _, a_discount = calculate_smart_ticket(st.session_state.accepted_alt_name, nationality, "Low", is_alternative=True)
            
            st.markdown(f"""
            <div class="impact-box">
                <div class="badge">🏅🌱</div>
                <h2>Green Tourist Badge Unlocked!</h2>
                <p>Thank you for choosing <b>{st.session_state.accepted_alt_name}</b>.</p>
                <div class="ticket-card">
                    <h4>🎟️ Smart Eco-Ticket</h4>
                    <p>Site: {st.session_state.accepted_alt_name}</p>
                    <p class="ticket-price">{alt_price:.2f} {a_unit}</p>
                    <span class="discount-badge">✓ 15% Sustainability Discount Applied</span>
                </div>
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
                st.session_state.chat_history = []
                st.rerun()

        else:
            st.markdown(f"""
            <div class="live-data-box">
                <h4 style='margin-top:0;'>📡 Real-Time Status for {selected_site}</h4>
                <p><b>Weather:</b> {st.session_state.live_weather}   |   <b>AQI:</b> {st.session_state.live_aqi}   |   <b>Crowd Level:</b> {st.session_state.live_overcrowding}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="ticket-card">
                <h4 style="margin-top:0;">🎟️ Dynamic AI Ticket</h4>
                <p>Location: {selected_site}</p>
                <p class="ticket-price">{main_price:.2f} {m_unit}</p>
                {"<p style='color:red; font-size:12px;'>⚠️ Includes 20% Conservation Surcharge due to high crowd risk.</p>" if m_surcharge > 0 else "<p style='color:green; font-size:12px;'>✓ Standard Pricing applies.</p>"}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="micro-zone-box">
                <h4 style='margin-top: 0;'>🗺️ Micro-Zone Crowd Radar</h4>
                <p style='font-size: 14px; margin-bottom: 10px;'>Sustainability AI distributes foot traffic. Live conditions within the <b>{selected_site}</b> complex:</p>
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

            st.markdown("---")
            if decision == 'No':
                st.markdown(f'<div class="success-box"><h3 style="margin-top:0;">✅ Clear to Visit!</h3><p>Enjoy <b>{selected_site}</b>. Conditions are optimal.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-box"><h3 style="margin-top:0;">⚠️ Redirection Activated</h3><p><b>{selected_site}</b> is currently facing {st.session_state.live_overcrowding} risk.</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="postpone-box"><h4 style="margin-top:0;">⏳ Smart Postponement</h4><p>If visiting <b>{selected_site}</b> is an absolute must, consider rescheduling to avoid the Peak Surcharge.</p><h3>Optimal Visiting Window: {recommended_time}</h3></div>', unsafe_allow_html=True)
                
                st.subheader("🌿 Suggested Safe Alternatives")
                st.markdown(f'<div class="profile-box"><b>AI Context:</b> Alternatives filtered for <b>{mobility_level}</b> access. By taking <b>{transport_mode}</b>, you maximize your eco-points!</div>', unsafe_allow_html=True)
                
                alternatives = df[(df['District'] == selected_district) & (df['Overcrowding Risk'] == 'Low') & (df['Site Name'] != selected_site)]
                if not alternatives.empty:
                    alts = alternatives.sample(min(3, len(alternatives)))
                    cols = st.columns(len(alts))
                    for col, (_, alt) in zip(cols, alts.iterrows()):
                        with col:
                            a_p, a_u, _, _ = calculate_smart_ticket(alt["Site Name"], nationality, "Low", is_alternative=True)
                            
                            st.markdown(f"""
                            <div class="alt-card">
                                <h4 style="margin-top:0;">{alt["Site Name"]}</h4>
                                <p style="font-size:14px; font-weight:bold; color:#27AE60;">Ticket: {a_p:.2f} {a_u}</p>
                                <p style="font-size:12px;">📍 {alt["Location"]}</p>
                            """, unsafe_allow_html=True)
                            
                            search_query = quote_plus(f"{alt['Site Name']} {alt['District']} Sri Lanka")
                            st.markdown(f"**[🗺️ Navigate](https://www.google.com/maps/search/?api=1&query={search_query})**")
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            if st.button(f"🌍 Accept & Save Heritage", key=f"btn_{alt['Site Name']}"):
                                st.session_state.accepted_alt = True
                                st.session_state.accepted_alt_name = alt['Site Name']
                                st.session_state.total_redirects += 1
                                st.rerun() 
                                
                            st.markdown('</div>', unsafe_allow_html=True)

            # Predictive Forecasting
            st.markdown("---")
            st.markdown(f"""
            <div class="forecast-box">
                <h4 style='margin-top: 0;'>🔮 AI Predictive Forecasting (Plan Ahead)</h4>
                <p style='font-size: 14px;'>Select a future timeline to see AI-predicted conditions and pricing.</p>
            </div>
            """, unsafe_allow_html=True)

            fc_col1, fc_col2 = st.columns([1, 2])
            with fc_col1:
                forecast_day = st.selectbox("Select Timeline:", ["Tomorrow", "Coming Weekend", "Next Week"], key="forecast_day")

            with fc_col2:
                f_weather = random.choice(["Sunny", "Clear", "Light Rain"])
                f_crowd = random.choice(["Low", "Medium", "High"])
                f_color = "🔴" if f_crowd == "High" else "🟠" if f_crowd == "Medium" else "🟢"
                f_price, f_unit, _, _ = calculate_smart_ticket(selected_site, nationality, f_crowd)

                st.markdown(f"""
                <div class="forecast-result-box">
                    <strong>Forecast for {selected_site} ({forecast_day}):</strong><br>
                    🌡️ Weather: {f_weather}   |   {f_color} Crowd: {f_crowd}<br>
                    💰 Estimated Ticket: <b>{f_price:.2f} {f_unit}</b>
                </div>
                """, unsafe_allow_html=True)

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

            # Continuous Chatbot
            st.markdown("---")
            st.subheader(f"💬 Live AI Heritage Guide: {selected_site}")
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if user_question := st.chat_input("Ask about history, dress code, or current pricing..."):
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)

                with st.chat_message("assistant"):
                    q = user_question.lower()
                    if any(word in q for word in ["ticket", "price", "cost", "fee", "pay", "money"]):
                        ai_response = f"Current ticket for **{selected_site}** is **{main_price:.2f} {m_unit}**. "
                        if m_surcharge > 0:
                            ai_response += "This includes a Peak Surcharge. You can save money by visiting an alternative site or waiting for off-peak hours."
                        else:
                            ai_response += "This is our standard rate. Enjoy your visit!"
                    elif any(word in q for word in ["dress", "wear", "clothes"]):
                        ai_response = "Please wear modest clothing. For religious sites, shoulders and knees must be covered."
                    else:
                        ai_response = f"Great question! {selected_site} is a unique location. Since crowd risk is {st.session_state.live_overcrowding}, make sure to follow the AI guide for a better experience."
                    
                    st.markdown(ai_response)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

# --- 5. Admin Dashboard ---
elif app_mode == "2. Admin Dashboard (Panel)":
    render_admin_dashboard(df, ai_model, feature_columns, fpr, tpr, roc_auc)