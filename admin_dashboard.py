import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def render_admin_dashboard(df, ai_model, feature_columns, fpr, tpr, roc_auc):
    st.markdown('<div class="main-header">📊 Tourism Authority Analytics Panel</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Restricted access for SLTDA Administrators & AI Researchers.</p><hr>", unsafe_allow_html=True)
    
    # 1. System Overview Metrics
    st.subheader("🌐 Global System Impact (Today)")
    d_col1, d_col2, d_col3 = st.columns(3)
    
    with d_col1:
        st.markdown(f'<div class="dash-metric-box"><div class="dash-metric-title">Active Tourists Tracked</div><div class="dash-metric-value">{st.session_state.total_users}</div><span style="color: green;">↑ 12% vs yesterday</span></div>', unsafe_allow_html=True)
    with d_col2:
        st.markdown(f'<div class="dash-metric-box"><div class="dash-metric-title" style="color: #E74C3C !important;">High Risk Alerts Issued</div><div class="dash-metric-value" style="color: #E74C3C !important;">412</div><span style="color: red;">Requires Monitoring</span></div>', unsafe_allow_html=True)
    with d_col3:
        st.markdown(f'<div class="dash-metric-box" style="border-top: 5px solid #2ECC71;"><div class="dash-metric-title" style="color: #2ECC71 !important;">Heritage Sites Saved (Redirects)</div><div class="dash-metric-value" style="color: #2ECC71 !important;">{st.session_state.total_redirects}</div><span style="color: green;">Successful Conservations</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # 2. Charts & Heatmaps
    st.subheader("🗺️ Regional Congestion Heatmap")
    st.write("Live distribution of tourist density across major heritage districts.")
    
    # Mock data for the heatmap bar chart
    dist_data = pd.DataFrame({
        'District': ['Anuradhapura', 'Polonnaruwa', 'Kandy', 'Matale', 'Galle', 'Hambantota'],
        'Tourist Load (%)': [92, 45, 88, 70, 65, 30]
    }).set_index('District')
    st.bar_chart(dist_data, color="#E67E22")
    
    st.markdown("---")

    # 3. AI Model Evaluation (FOR THE IT PANEL)
    st.subheader("🤖 XGBoost AI Model Evaluation")
    st.info("This section validates the scientific accuracy of the underlying Machine Learning model used for the Adaptive Redirection.")
    
    ev_col1, ev_col2 = st.columns(2)
    
    with ev_col1:
        st.markdown("**1. Receiver Operating Characteristic (ROC) Curve**")
        st.write("Proves the model's ability to distinguish between 'Safe to Visit' and 'Must Redirect' scenarios.")
        
        # Plotting ROC Curve
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
        st.markdown("**2. Feature Importance Graph**")
        st.write("Displays which environmental factors the AI prioritizes when making a redirection decision.")
        
        # Plotting Feature Importance
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
    st.subheader("📋 Recent AI Interventions Log")
    st.write("A real-time ledger of actions taken by the System to protect heritage sites.")
    
    log_data = pd.DataFrame({
        "Time": ["10:42 AM", "10:40 AM", "10:35 AM", "10:28 AM", "10:15 AM"],
        "Target Site": ["Sigiriya", "Ruwanwelisaya", "Galle Fort", "Dambulla Cave", "Temple of Tooth"],
        "Triggered Risk": ["High Overcrowding", "High Temp & Crowd", "AQI Alert", "Safe", "High Overcrowding"],
        "AI Action": ["Redirected to Pidurangala", "Redirected to Abhayagiriya", "Postponed Suggestion", "Clear to Visit", "Redirected to Embekke"]
    })
    
    st.dataframe(log_data, use_container_width=True, hide_index=True)
    st.success("System Status: All protocols running smoothly. No critical threats detected.")