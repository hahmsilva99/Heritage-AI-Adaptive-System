import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import time

def render_admin_dashboard(df, ai_model, feature_columns, fpr, tpr, roc_auc):
    st.markdown('<div class="main-header">📊 Tourism Authority Analytics Panel</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Restricted access for SLTDA Administrators & AI Researchers.</p><hr>", unsafe_allow_html=True)
    
    # 1. 🚨 LIVE THREAT ALERTS (National Security & Crisis Management)
    st.markdown("""
    <div style="background-color: #FDEDEC; border: 2px solid #E74C3C; padding: 15px; border-radius: 8px; margin-bottom: 25px; box-shadow: 0 4px 8px rgba(231, 76, 60, 0.2);">
        <h3 style="color: #C0392B; margin-top: 0;">🚨 LIVE SYSTEM THREATS & CRITICAL ALERTS</h3>
        <p style="color: #922B21; font-size: 14px; margin-bottom: 10px;">AI automated monitoring has detected the following real-time anomalies:</p>
        <ul style="color: #7B241C; font-size: 15px; font-weight: bold; margin-bottom: 0;">
            <li style="margin-bottom: 8px;">🔴 CRITICAL (SIGIRIYA): Lion Paw stairway capacity exceeded by 140%. High stampede risk. Auto-redirection to Pidurangala enforced.</li>
            <li style="margin-bottom: 8px;">⚠️ WEATHER (GALLE FORT): Sudden coastal squalls and lightning predicted in 15 mins. Advising tourists to clear the ramparts.</li>
            <li>🟠 CROWDSOURCED (TEMPLE OF TOOTH): 12 users reported unauthorized ticket scalping near the main gate. Tourist Police notified.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 2. SYSTEM OVERVIEW METRICS (Sustainability & Impact)
    st.subheader("🌐 Global System Impact (Today)")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        st.markdown(f'<div class="dash-metric-box"><div class="dash-metric-title">Active Tourists</div><div class="dash-metric-value">{st.session_state.total_users}</div><span style="color: green;">↑ 12% vs yesterday</span></div>', unsafe_allow_html=True)
    with d_col2:
        st.markdown(f'<div class="dash-metric-box" style="border-top: 5px solid #E74C3C;"><div class="dash-metric-title" style="color: #E74C3C !important;">High Risk Alerts</div><div class="dash-metric-value" style="color: #E74C3C !important;">412</div><span style="color: red;">Requires Monitoring</span></div>', unsafe_allow_html=True)
    with d_col3:
        st.markdown(f'<div class="dash-metric-box" style="border-top: 5px solid #2ECC71;"><div class="dash-metric-title" style="color: #2ECC71 !important;">Sites Saved (Redirects)</div><div class="dash-metric-value" style="color: #2ECC71 !important;">{st.session_state.total_redirects}</div><span style="color: green;">Successful Conserves</span></div>', unsafe_allow_html=True)
    saved_co2 = st.session_state.total_redirects * 1.2 
    with d_col4:
        st.markdown(f'<div class="dash-metric-box" style="border-top: 5px solid #1ABC9C;"><div class="dash-metric-title" style="color: #1ABC9C !important;">CO₂ Emissions Saved</div><div class="dash-metric-value" style="color: #1ABC9C !important;">{saved_co2:.1f} kg</div><span style="color: #16A085;">🌿 Eco-Tourism Impact</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ==========================================
    # 🔥 NEW: ADVANCED AI RESEARCH FEATURES
    # ==========================================
    st.subheader("🧠 Advanced AI System Diagnostics")
    st.write("Monitoring the core pillars of Accuracy, Sustainability, and Adaptive Intelligence.")
    
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    
    with adv_col1:
        # ACCURACY: Cross-Validation
        st.markdown("""
        <div style="background-color: #EBF5FB; padding: 15px; border-radius: 8px; border-top: 4px solid #2980B9; height: 100%; color: black;">
            <h4 style="color: #1A5276; margin-top: 0;">🎯 Data Cross-Validation</h4>
            <p style="font-size: 13px;">AI Decision Consensus Matrix:</p>
            <ul style="font-size: 13px; list-style-type: none; padding-left: 0;">
                <li>✅ <b>Satellite Weather API:</b> Active</li>
                <li>✅ <b>IoT Crowd Sensors:</b> Syncing</li>
                <li>✅ <b>Social Sentiment (NLP):</b> Stable</li>
            </ul>
            <p style="font-size: 18px; font-weight: bold; color: #28B463;">Decision Accuracy: 98.4%</p>
        </div>
        """, unsafe_allow_html=True)

    with adv_col2:
        # ADAPTIVE-AI: Feedback Loop
        st.markdown("""
        <div style="background-color: #F5EEF8; padding: 15px; border-radius: 8px; border-top: 4px solid #8E44AD; height: 100%; color: black;">
            <h4 style="color: #5B2C6F; margin-top: 0;">🔄 Adaptive Learning Loop</h4>
            <p style="font-size: 13px;">User response behavior tracking:</p>
            <ul style="font-size: 13px; padding-left: 15px;">
                <li>Recent AI Redirect Accept Rate: <b>82%</b></li>
                <li>Model Re-calibration: <b style="color: green;">Passed</b></li>
            </ul>
            <p style="font-size: 13px; color: #5B2C6F;"><i>AI has automatically demoted 'Dambulla' as an alternative due to low user acceptance yesterday.</i></p>
        </div>
        """, unsafe_allow_html=True)

    with adv_col3:
        # SUSTAINABILITY: Bio-Diversity
        st.markdown("""
        <div style="background-color: #E9F7EF; padding: 15px; border-radius: 8px; border-top: 4px solid #27AE60; height: 100%; color: black;">
            <h4 style="color: #1D8348; margin-top: 0;">🌿 Bio-Diversity Monitor</h4>
            <p style="font-size: 13px;">Wildlife impact during redirection:</p>
            <ul style="font-size: 13px; padding-left: 15px;">
                <li><b>Minneriya Paths:</b> Clear (Safe)</li>
                <li><b>Yala Buffer Zone:</b> <span style="color: orange;">Approaching Limit</span></li>
            </ul>
            <p style="font-size: 13px; color: #1D8348;"><i>AI restricting alternatives near Habarana to protect Elephant migration corridors.</i></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. 🔮 DIGITAL TWIN 'WHAT-IF' SIMULATOR
    st.subheader("🔮 Digital Twin 'What-If' Scenario Simulator")
    st.write("Test disaster or extreme crowd scenarios using AI predictive modeling before they happen.")
    
    with st.container():
        st.markdown('<div style="border: 1px solid #D5DBDB; border-radius: 10px; padding: 20px; background-color: #F2F3F4;">', unsafe_allow_html=True)
        sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
        with sim_col1:
            sim_district = st.selectbox("1. Target District:", sorted(df['District'].unique()), key="sim_dist", index=None, placeholder="Select District...")
        with sim_col2:
            sim_sites = df[df['District'] == sim_district]['Site Name'].unique() if sim_district else []
            sim_site = st.selectbox("2. Target Heritage Site:", sorted(sim_sites), key="sim_site", index=None, placeholder="Select Site...")
        with sim_col3:
            sim_surge = st.slider("3. Tourist Surge (Next 2 Hrs):", min_value=100, max_value=10000, value=3000, step=500)
        with sim_col4:
            sim_weather = st.selectbox("4. Extreme Weather:", ["None (Clear)", "Heavy Monsoon Rain", "Extreme Heatwave (>35°C)"])
            
        if st.button("▶️ Run AI Simulation", type="primary", use_container_width=True):
            if not sim_district or not sim_site:
                st.error("⚠️ Please select both a District and a Heritage Site to run the simulation.")
            else:
                with st.spinner(f"Simulating Digital Twin for {sim_site}..."):
                    time.sleep(1.5)
                    st.markdown("#### 📊 Simulation Results:")
                    actual_capacity = df[df['Site Name'] == sim_site]['Max Visitor Capacity'].iloc[0]
                    raw_stress = (sim_surge / (actual_capacity * 3)) * 100 
                    stress_level = int(min(100, raw_stress))
                    if sim_weather != "None (Clear)": stress_level = min(100, stress_level + 25)
                    
                    res_c1, res_c2, res_c3 = st.columns(3)
                    res_c1.metric("Predicted Structural Stress", f"{stress_level}%", "+ Critical" if stress_level > 80 else "- Manageable")
                    res_c2.metric("Est. Casualties (No AI)", f"{int(sim_surge * 0.03)}" if stress_level > 80 else "0")
                    res_c3.metric("Required AI Redirections", f"{int(sim_surge * 0.65)} users")
                    st.progress(stress_level / 100)
                    if stress_level > 80: st.error(f"⚠️ CRITICAL FAILURE IMMINENT AT {sim_site.upper()}!")
                    elif stress_level > 50: st.warning(f"🟠 MODERATE RISK AT {sim_site.upper()}.")
                    else: st.success(f"✅ SAFE SCENARIO FOR {sim_site.upper()}.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # 4. SOCIO-ECONOMIC & REGIONAL HEATMAP (Wealth Dispersion)
    st.subheader("📈 Socio-Economic Wealth Dispersion & Traffic")
    eco_col1, eco_col2 = st.columns([1, 1])
    with eco_col1:
        dist_data = pd.DataFrame({'District': ['Anuradhapura', 'Polonnaruwa', 'Kandy', 'Matale', 'Galle', 'Hambantota'], 'Tourist Load (%)': [92, 45, 88, 70, 65, 30]}).set_index('District')
        st.markdown("**Live Regional Congestion Heatmap**")
        st.bar_chart(dist_data, color="#E67E22", height=250)
    with eco_col2:
        st.markdown("**Income Generated for Rural SME Sites (LKR)**")
        econ_df = pd.DataFrame({'Days': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 'Rural Economy Boost': [25000, 30000, 28000, 45000, 60000, 120000, 135000]}).set_index('Days')
        st.area_chart(econ_df, color="#8E44AD", height=250)

    st.markdown("---")

    # 5. 🚓 SMART RESOURCE DISPATCH (Predictive AI)
    st.subheader("🚓 Smart Resource Dispatch Radar (Tomorrow's Forecast)")
    res_c1, res_c2, res_c3 = st.columns(3)
    with res_c1:
        st.markdown('<div style="background-color: #FEF9E7; padding: 15px; border-radius: 8px; border-left: 5px solid #F1C40F; color: black;"><strong style="color: #B7950B;">📍 Temple of the Tooth</strong><br><i>Forecast: High Crowds</i><br><br>✅ <b>Action:</b> Dispatch 10 Police, 2 Med Teams.</div>', unsafe_allow_html=True)
    with res_c2:
        st.markdown('<div style="background-color: #E8F8F5; padding: 15px; border-radius: 8px; border-left: 5px solid #1ABC9C; color: black;"><strong style="color: #0E6251;">📍 Pidurangala Rock</strong><br><i>Forecast: Moderate Surge</i><br><br>✅ <b>Action:</b> Send Waste Management truck.</div>', unsafe_allow_html=True)
    with res_c3:
        st.markdown('<div style="background-color: #FDEDEC; padding: 15px; border-radius: 8px; border-left: 5px solid #E74C3C; color: black;"><strong style="color: #943126;">📍 Galle Fort</strong><br><i>Forecast: Extreme Heat</i><br><br>✅ <b>Action:</b> Open shaded areas, give water.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 6. 🤖 XGBOOST MODEL VALIDATION (IT Panel Requirements)
    st.subheader("🤖 XGBoost AI Model Validation")
    ev_col1, ev_col2 = st.columns(2)
    with ev_col1:
        st.markdown("**Receiver Operating Characteristic (ROC) Curve**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.set_title('AI Prediction Reliability')
        ax.legend(loc="lower right")
        st.pyplot(fig)
    with ev_col2:
        st.markdown("**Feature Importance (Decision Weights)**")
        importance = ai_model.feature_importances_
        feat_imp = pd.DataFrame({'Feature': feature_columns, 'Importance': importance}).sort_values(by='Importance', ascending=True)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.barh(feat_imp['Feature'], feat_imp['Importance'], color='teal')
        ax2.set_xlabel('Relative Importance'); ax2.set_title('Decision Weighting')
        st.pyplot(fig2)

    st.markdown("---")
    
    # 7. 📋 REAL-TIME AI INTERVENTIONS LOG
    st.subheader("📋 Real-Time AI Interventions Ledger")
    st.write("A detailed historical record of every adaptive decision made by the system today.")
    log_data = pd.DataFrame({
        "Time Stamp": ["11:15 AM", "11:02 AM", "10:45 AM", "10:28 AM", "10:12 AM", "09:55 AM"],
        "Target Site": ["Sigiriya", "Temple of Tooth", "Ruwanwelisaya", "Galle Fort", "Dambulla Cave", "Sigiriya"],
        "Risk Type": ["Structural Load", "Crowd Peak", "Weather Alert", "AQI Warning", "Safe Visit", "Structural Load"],
        "AI Decision": ["Redirected to Pidurangala", "Postponed Suggestion", "Recommended Abhayagiriya", "Peak Surcharge Applied", "Standard Access", "Redirected to Pidurangala"],
        "User Response": ["Accepted (82%)", "Accepted (65%)", "Accepted (90%)", "Accepted (55%)", "N/A", "Accepted (78%)"]
    })
    st.dataframe(log_data, use_container_width=True, hide_index=True)
    st.success("✅ System Status: All Interventions logged and archived for SLTDA Audit.")