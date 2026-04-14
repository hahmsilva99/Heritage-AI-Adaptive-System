# 🏛️ Eco-Adaptive Heritage Explorer: AI-Driven Sustainable Tourism System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![Machine Learning](https://img.shields.io/badge/AI-XGBoost-28B463)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

## 📌 Project Overview
The **Eco-Adaptive Heritage Explorer** is an advanced, AI-powered load-balancing and predictive management system designed for the Sri Lanka Tourism Development Authority (SLTDA). It aims to mitigate "Overtourism" at historical heritage sites by using Machine Learning to dynamically redirect tourists to alternative rural sites based on real-time crowd, weather, and structural stress data.

This project bridges the gap between **Heritage Conservation** and **Tourism Economics** by employing adaptive AI, dynamic pricing, and digital twin simulations.

## 🚀 Key Features

### 👤 1. Tourist Explorer (User Interface)
* **Smart Eco-Profiling:** Context-aware routing based on mobility needs, nationality, and transport mode.
* **AI Redirection Engine:** Suggests low-risk, high-reward alternative sites when primary destinations are critically crowded.
* **Dynamic Smart Ticketing:** Automatically applies "Peak Surcharges" for crowded sites and "Sustainability Discounts" (15%) for alternative sites.
* **Context-Aware NLP Chatbot:** A built-in virtual guide providing location-specific history, dress codes, and ticket pricing without requiring external API dependencies.
* **Gamification & Impact Tracking:** Rewards users with "Green Tourist Badges" and tracks reduced CO₂ emissions and local economic boosts.

### 🛡️ 2. SLTDA Admin Dashboard (Management Interface)
* **Digital Twin 'What-If' Simulator:** Allows authorities to simulate disaster scenarios (e.g., Extreme Weather + High Surge) to predict structural stress, casualty risks, and required AI redirections.
* **Live Threat Alerts:** Real-time monitoring of crowd panic, unauthorized activities, and structural overload.
* **Predictive Resource Dispatch:** Auto-generates tomorrow's resource needs (e.g., deploying Tourist Police or Medical Teams based on XGBoost forecasts).
* **Model Validation Panel:** Displays ROC Curves and Feature Importance graphs to validate AI decision-making accuracy.
* **Socio-Economic Wealth Dispersion:** Visualizes how redirecting traffic boosts rural SME (Small & Medium Enterprise) economies.

## 🧠 Technical Architecture & AI Model
* **Algorithm:** XGBoost Classifier (Extreme Gradient Boosting) optimized for categorical multi-variable inputs.
* **Data Balancing:** SMOTE (Synthetic Minority Over-sampling Technique) applied to handle imbalanced redirection data.
* **Evaluation Metrics:** ROC-AUC tracking and Live Feature Importance Weighting.
* **Frontend/Backend Routing:** Modular Streamlit architecture separating client views (`app.py`) from administrative logic (`admin_dashboard.py`).

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/eco-adaptive-heritage-ai.git](https://github.com/your-username/eco-adaptive-heritage-ai.git)
   cd eco-adaptive-heritage-ai