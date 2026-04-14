import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

def render_admin_dashboard(df, ai_model, feature_columns, fpr, tpr, roc_auc):
    st.markdown('<div class="main-header">📊 Tourism Authority Analytics Panel</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Restricted access for SLTDA Administrators & AI Researchers.</p><hr>", unsafe_allow_html=True)
    
    # 1. System Overview Metrics
    st.subheader("🌐 Global System Impact (Today)")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    
    with d_col1:
        st.markdown(f'<div class="dash-metric-box"><div class="dash-metric-title">Active Tourists</div><div class="dash-metric-value">{st.session_state.total_users}</div><span style="color: green;">↑ 12% vs yesterday</span></div>', unsafe_allow_html=True)
    with d_col2:
        st.markdown(f'<div class="dash-metric-box" style="border-top: 5px solid #E74C3C;"><div class="dash-metric-title" style="color: #E74C3C !important;">High Risk Alerts</div><div class="dash-metric-value" style="color: #E74C3C !important;">412</div><span style="color: red;">Requires Monitoring</span></div>', unsafe_allow_html=True)
    with d_col3:
        st.markdown(f'<div class="dash-metric-box" style="border-top: 5px solid #2ECC71;"><div class="dash-metric-title" style="color: #2ECC71 !important;">Sites Saved (Redirects)</div><div class="dash-metric-value" style="color: #2ECC71 !important;">{st.session_state.total_redirects}</div><span style="color: green;">Successful Conserves</span></div>', unsafe_allow_html=True)
    
    # 🔥 NEW FEATURE: Carbon Footprint Saver
    saved_co2 = st.session_state.total_redirects * 1.2 # Assuming 1.2kg of CO2 saved per redirect (less traffic/idling)
    with d_col4:
        st.markdown(f'<div class="dash-metric-box" style="border-top: 5px solid #1ABC9C;"><div class="dash-metric-title" style="color: #1ABC9C !important;">CO₂ Emissions Saved</div><div class="dash-metric-value" style="color: #1ABC9C !important;">{saved_co2:.1f} kg</div><span style="color: #16A085;">🌿 Eco-Tourism Impact</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # 🔥 NEW FEATURE: Socio-Economic & Regional Heatmap
    st.subheader("📈 Socio-Economic Wealth Dispersion & Traffic")
    st.write("Visualizing how AI redirection distributes tourism revenue to rural/alternative sites while reducing main-site traffic.")
    
    eco_col1, eco_col2 = st.columns([1, 1])
    
    with eco_col1:
        # Existing Heatmap
        dist_data = pd.DataFrame({
            'District': ['Anuradhapura', 'Polonnaruwa', 'Kandy', 'Matale', 'Galle', 'Hambantota'],
            'Tourist Load (%)': [92, 45, 88, 70, 65, 30]
        }).set_index('District')
        st.markdown("**Live Regional Congestion Heatmap**")
        st.bar_chart(dist_data, color="#E67E22", height=250)

    with eco_col2:
        # New Economic Dispersion Chart
        st.markdown("**Income Generated for Alternative Sites (LKR)**")
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        # Mock data showing rising income for alternative sites
        alt_income = [25000, 30000, 28000, 45000, 60000, 120000, 135000] 
        main_income_reduction = [-10000, -15000, -12000, -25000, -35000, -60000, -75000] # Kept as info
        
        econ_df = pd.DataFrame({'Days': days, 'Rural Economy Boost': alt_income}).set_index('Days')
        st.area_chart(econ_df, color="#8E44AD", height=250)

    st.markdown("---")

    # 🔥 NEW FEATURE: Predictive Resource Dispatch AI
    st.subheader("🚓 Smart Resource Dispatch Radar (Tomorrow's Forecast)")
    st.info("AI predicts upcoming congestion and automatically recommends resource allocations for SLTDA authorities.")
    
    res_c1, res_c2, res_c3 = st.columns(3)
    with res_c1:
        st.markdown("""
        <div style="background-color: #FEF9E7; padding: 15px; border-radius: 8px; border-left: 5px solid #F1C40F; color: black;">
            <strong style="color: #B7950B;">📍 Temple of the Tooth (Kandy)</strong><br>
            <i>Forecast: High Crowds (Poya Day)</i><br><br>
            ✅ <b>Action Required:</b><br>
            • Dispatch 10 Extra Tourist Police.<br>
            • Deploy 2 additional medical teams.
        </div>
        """, unsafe_allow_html=True)
        
    with res_c2:
        st.markdown("""
        <div style="background-color: #E8F8F5; padding: 15px; border-radius: 8px; border-left: 5px solid #1ABC9C; color: black;">
            <strong style="color: #0E6251;">📍 Pidurangala Rock (Matale)</strong><br>
            <i>Forecast: Moderate Surge (AI Redirects)</i><br><br>
            ✅ <b>Action Required:</b><br>
            • Notify local certified guides.<br>
            • Send 1 extra Waste Management truck.
        </div>
        """, unsafe_allow_html=True)
        
    with res_c3:
        st.markdown("""
        <div style="background-color: #FDEDEC; padding: 15px; border-radius: 8px; border-left: 5px solid #E74C3C; color: black;">
            <strong style="color: #943126;">📍 Galle Fort (Galle)</strong><br>
            <i>Forecast: Extreme Heat & High UV</i><br><br>
            ✅ <b>Action Required:</b><br>
            • Open all shaded resting areas.<br>
            • Distribute free drinking water points.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # 3. AI Model Evaluation (FOR THE IT PANEL)
    st.subheader("🤖 XGBoost AI Model Validation")
    ev_col1, ev_col2 = st.columns(2)
    
    with ev_col1:
        st.markdown("**Receiver Operating Characteristic (ROC) Curve**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('AI Prediction Reliability')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        
    with ev_col2:
        st.markdown("**Feature Importance (Decision Weights)**")
        importance = ai_model.feature_importances_
        feat_imp = pd.DataFrame({'Feature': feature_columns, 'Importance': importance})
        feat_imp = feat_imp.sort_values(by='Importance', ascending=True)
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.barh(feat_imp['Feature'], feat_imp['Importance'], color='teal')
        ax2.set_xlabel('Relative Importance')
        ax2.set_title('Decision Weighting')
        st.pyplot(fig2)

    st.markdown("---")
    
    # 4. Recent AI Decisions Log Table
    st.subheader("📋 Real-Time AI Interventions Ledger")
    log_data = pd.DataFrame({
        "Time": ["10:42 AM", "10:40 AM", "10:35 AM", "10:28 AM", "10:15 AM"],
        "Target Site": ["Sigiriya", "Ruwanwelisaya", "Galle Fort", "Dambulla Cave", "Temple of Tooth"],
        "Triggered Risk": ["High Overcrowding", "High Temp & Crowd", "AQI Alert", "Safe", "High Overcrowding"],
        "AI Action": ["Redirected to Pidurangala", "Redirected to Abhayagiriya", "Postponed Suggestion", "Clear to Visit", "Redirected to Embekke"]
    })
    st.dataframe(log_data, use_container_width=True, hide_index=True)
    st.success("System Status: Eco-Adaptive Routing and Dispatch Protocols running flawlessly.")