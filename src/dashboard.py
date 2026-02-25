# import streamlit as st
# import joblib
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from preprocess import clean_text


# MODEL_PATH = 'D:/VS Code/Python/CyberSec_Threat_Detection/models/email_model.pkl'
# VECTORIZER_PATH = 'D:/VS Code/Python/CyberSec_Threat_Detection/models/vectorizer.pkl'

# # 1. Page Configuration
# st.set_page_config(
#     page_title="AI Cyber Shield Dashboard",
#     page_icon="🛡️",
#     layout="wide"
# )

# # 2. Load the "Brain" (Model and Vectorizer)
# @st.cache_resource
# def load_assets():
#     if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
#         return None, None
#     model = joblib.load(MODEL_PATH)
#     vectorizer = joblib.load(VECTORIZER_PATH)
#     return model, vectorizer

# model, vectorizer = load_assets()

# # --- HEADER SECTION ---
# st.title("🛡️ AI-Powered Cybersecurity Threat Detection")
# st.markdown("""
#     This dashboard provides a real-time interface for your **Email Threat Detection System**. 
#     It uses a **Random Forest** model to identify malicious phishing attempts and automate prevention.
# """)

# if model is None:
#     st.error("🚨 Models not found! Please run `python src/train.py` first to train your AI.")
#     st.stop()

# # --- SIDEBAR ---
# st.sidebar.header("System Configuration")
# st.sidebar.info("Model: **Random Forest**")
# st.sidebar.info("Status: **Active**")
# st.sidebar.divider()
# st.sidebar.write("Developed by Vicky Patil")

# # --- MAIN DASHBOARD LAYOUT ---
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.subheader("🔍 Live Threat Analyzer")
#     st.write("Paste the content of a suspicious email below to check for threats.")
    
#     email_input = st.text_area("Email Content:", placeholder="Enter email text here...", height=200)
    
#     if st.button("Run Security Scan"):
#         if email_input.strip():
#             # Process the input
#             cleaned = clean_text(email_input)
#             vectorized = vectorizer.transform([cleaned])
            
#             # Prediction and Probability
#             prediction = model.predict(vectorized)[0]
#             probability = model.predict_proba(vectorized)
            
#             st.divider()
#             if prediction == 1:
#                 st.error(f"### 🚨 ALERT: Phishing Detected!")
#                 st.warning(f"Confidence Score: {probability[0][1]*100:.2f}%")
#                 st.write("**Recommended Action:** Quarantine email and block sender.")
#             else:
#                 st.success(f"### ✅ Result: Safe Email")
#                 st.info(f"Confidence Score: {probability[0][0]*100:.2f}%")
#         else:
#             st.warning("Please enter email text to analyze.")

# with col2:
#     st.subheader("📊 Model Metrics")
#     # Visualization of the confidence
#     if 'probability' in locals():
#         fig, ax = plt.subplots()
#         labels = ['Safe', 'Phishing']
#         counts = [probability[0][0], probability[0][1]]
#         ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
#         st.pyplot(fig)
#     else:
#         st.write("Run a scan to see the probability breakdown.")

# # --- FOOTER SECTION ---
# st.divider()
# st.subheader("📈 System Logs (Mockup)")
# log_data = {
#     "Timestamp": ["2026-02-21 12:01", "2026-02-21 12:05", "2026-02-21 12:15"],
#     "Event": ["Inbound Scan", "Phishing Blocked", "Inbound Scan"],
#     "Status": ["Clean", "🚨 BLOCKED", "Clean"],
#     "Source": ["hr@company.com", "verify-acc-32@scam.net", "newsletter@tech.com"]
# }
# st.table(pd.DataFrame(log_data))



#------------------------------------------

# import streamlit as st
# import joblib
# import os
# import pandas as pd
# import time
# from preprocess import clean_text
# from streamlit_autorefresh import st_autorefresh

# # 1. Page Configuration
# st.set_page_config(page_title="AI Cyber Shield SOC", page_icon="🛡️", layout="wide")

# # 2. Load the "Brain"
# @st.cache_resource
# def load_assets():
#     if not os.path.exists('models/email_model.pkl') or not os.path.exists('models/vectorizer.pkl'):
#         return None, None
#     return joblib.load('models/email_model.pkl'), joblib.load('models/vectorizer.pkl')

# model, vectorizer = load_assets()

# # --- TABS FOR DIFFERENT VIEWS ---
# tab1, tab2 = st.tabs(["🔍 Manual Threat Analyzer", "📡 Live Security Monitor"])

# # --- TAB 1: MANUAL ANALYSIS ---
# with tab1:
#     st.header("Manual Email Analysis")
#     #auto refresh code
#     email_input = st.text_area("Paste email content here:", height=200)
    
#     if st.button("Scan for Threats"):
#         if email_input.strip() and model:
#             cleaned = clean_text(email_input)
#             vec = vectorizer.transform([cleaned])
#             pred = model.predict(vec)[0]
#             prob = model.predict_proba(vec)[0]
            
#             if pred == 1:
#                 st.error(f"🚨 ALERT: Phishing Detected! (Confidence: {prob[1]*100:.2f}%)")
#             else:
#                 st.success(f"✅ Safe Email (Confidence: {prob[0]*100:.2f}%)")

# # --- TAB 2: LIVE MONITORING ---
# with tab2:
#     st.header("Real-Time Security Feed")
#     st.write("This feed updates automatically as the background monitor detects threats.")
    
#     # Placeholder for the live table
#     log_container = st.empty()
    
#     # Path to the log file shared with monitor.py
#     LOG_FILE = 'D:/VS Code/Python/CyberSec_Threat_Detection/data/security_logs.csv'

#     def get_logs():
#         if os.path.exists(LOG_FILE):
#             df = pd.read_csv(LOG_FILE, names=["timestamp", "sender", "subject", "status", "confidence"])
#             return df.iloc[::-1]
#         return pd.DataFrame(columns=["timestamp", "sender", "subject", "status", "confidence"])

#     # This loop keeps the tab "alive" and updating
#     # In a real app, use st.empty() to refresh specific parts
#     if st.checkbox("Enable Auto-Refresh", value=True):
#         while True:
#             current_logs = get_logs()
#             with log_container.container():
#                 st.dataframe(current_logs, use_container_width=True)
                
#                 # Simple Metrics
#                 if not current_logs.empty:
#                     threat_count = current_logs["status"].str.contains("BLOCKED", na=False).sum()
#                     st.metric("Threats Blocked Today", threat_count)
            
#             time.sleep(5) # Refresh every 5 seconds
#             st.rerun() # Forces the dashboard to update with new data


#------------------------------------------------


import streamlit as st
import joblib
import os
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from preprocess import clean_text
from streamlit_autorefresh import st_autorefresh
import base64

# 1. Load the AI Model and Vectorizer

MODEL_PATH = 'D:/VS Code/Python/CyberSec_Threat_Detection/models/email_model.pkl'
VECTORIZER_PATH = 'D:/VS Code/Python/CyberSec_Threat_Detection/models/vectorizer.pkl'

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f"""
                <style>
                .stApp {{
                    background-image: linear-gradient(rgba(0, 0, 0, 0.9), rgba(0, 0, 255, 0.5)), url("data:image/png;base64,{bin_str}");
                    background-size: cover;
                }}
                </style>
                """
    st.markdown(page_bg_img, unsafe_allow_html=True)


image_path = "D:/VS Code/Python/CyberSec_Threat_Detection/cybersec_background_image.jpg" 

set_png_as_page_bg(image_path)


# 1. Page Configuration
st.set_page_config(
    page_title="AI Cyber Shield SOC", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 7rem;
        color: #00ff00;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px #000000;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00ff00;
    }
    .threat-alert {
        background-color: #ff000020;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ff0000;
    }
    .safe-alert {
        background-color: #00ff0020;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #00ff00;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load the "Brain"
@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

model, vectorizer = load_assets()

# 4. Initialize session state
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# 5. Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.markdown("## **AI Cyber Shield**")
    st.markdown("---")
    
    st.markdown("### System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🟢 **Model**")
        st.markdown("🟢 **Monitor**")
    with col2:
        st.markdown("Active" if model else "Inactive")
        st.markdown("Running")
    
    st.markdown("---")
    st.markdown("### Settings")
    refresh_rate = st.slider("Refresh Rate (seconds)", min_value=2, max_value=30, value=5)
    max_logs = st.slider("Max Logs to Display", min_value=10, max_value=100, value=50)
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    **AI-Powered Cybersecurity Threat Detection**
    
    - Model: Random Forest
    - Version: 2.0
    - Developer: Vicky Patil
    """)

# Main Header
st.markdown('<h1 class="main-header">🛡️ AI Cyber Threat Detection and Prevention System</h1>', unsafe_allow_html=True)

if model is None:
    st.error("🚨 Models not found! Please run `python src/train.py` first to train your AI.")
    st.stop()

# --- TABS FOR DIFFERENT VIEWS ---
tab1, tab2, tab3 = st.tabs(["🔍 Manual Analyzer", "📡 Live Monitor", "📊 Analytics Dashboard"])

# --- TAB 1: MANUAL ANALYSIS ---
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Email Analysis Tool")
        email_input = st.text_area(
            "Paste email content for analysis:", 
            height=200,
            placeholder="Enter suspicious email content here..."
        )
        
        col_scan, col_clear = st.columns([1, 5])
        with col_scan:
            scan_button = st.button("🔍 Scan Email", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                email_input = ""
                st.rerun()
        
        if scan_button and email_input.strip() and model:
            with st.spinner("Analyzing email content..."):
                # Process the input
                cleaned = clean_text(email_input)
                vectorized = vectorizer.transform([cleaned])
                
                # Prediction and Probability
                prediction = model.predict(vectorized)[0]
                probability = model.predict_proba(vectorized)[0]
                
                # Add to history
                st.session_state.scan_history.append({
                    'timestamp': datetime.now(),
                    'content': email_input[:50] + "...",
                    'prediction': 'Phishing' if prediction == 1 else 'Safe',
                    'confidence': probability[1] if prediction == 1 else probability[0]
                })
                
                st.markdown("---")
                
                # Display results with enhanced visuals
                if prediction == 1:
                    st.markdown('<div class="threat-alert">', unsafe_allow_html=True)
                    col_alert1, col_alert2 = st.columns([1, 2])
                    with col_alert1:
                        st.markdown("# 🚨")
                    with col_alert2:
                        st.markdown("### **PHISHING DETECTED!**")
                        st.markdown(f"**Confidence:** {probability[1]*100:.2f}%")
                        st.progress(float(probability[1]), text="Threat Level")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Recommended actions
                    st.warning("**Recommended Actions:**")
                    st.markdown("""
                    - ✅ Quarantine the email
                    - 🚫 Block sender domain
                    - 📢 Alert security team
                    - 🔒 Update email filters
                    """)
                else:
                    st.markdown('<div class="safe-alert">', unsafe_allow_html=True)
                    col_alert1, col_alert2 = st.columns([1, 2])
                    with col_alert1:
                        st.markdown("# ✅")
                    with col_alert2:
                        st.markdown("### **EMAIL IS SAFE**")
                        st.markdown(f"**Confidence:** {probability[0]*100:.2f}%")
                        st.progress(float(probability[0]), text="Safety Level")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        elif scan_button and not email_input.strip():
            st.warning("⚠️ Please enter email content to analyze.")
    
    with col2:
        st.markdown("### 📊 Analysis History")
        if st.session_state.scan_history:
            history_df = pd.DataFrame(st.session_state.scan_history[-10:])
            
            # Color-coded history
            for _, row in history_df.iterrows():
                color = "🔴" if row['prediction'] == 'Phishing' else "🟢"
                st.markdown(f"{color} **{row['timestamp'].strftime('%H:%M:%S')}** - {row['prediction']} ({row['confidence']*100:.1f}%)")
            
            # Summary metrics
            total_scans = len(st.session_state.scan_history)
            threats_found = sum(1 for x in st.session_state.scan_history if x['prediction'] == 'Phishing')
            
            st.markdown("---")
            st.markdown("### 📈 Scan Summary")
            st.metric("Total Scans", total_scans)
            st.metric("Threats Detected", threats_found, delta=f"{threats_found/total_scans*100:.1f}%" if total_scans > 0 else "0%")
        else:
            st.info("No scan history yet. Run an analysis to see history here.")

# --- TAB 2: LIVE MONITORING ---
with tab2:
    # Auto-refresh control
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)
    if auto_refresh:
        st_autorefresh(interval=refresh_rate * 1000, key="auto_refresh")
    
    # Path to the log file
    LOG_FILE = 'D:/VS Code/Python/CyberSec_Threat_Detection/data/security_logs.csv'

    def get_logs():
        if os.path.exists(LOG_FILE):
            try:
                df = pd.read_csv(LOG_FILE, names=["timestamp", "sender", "subject", "status", "confidence"])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.sort_values('timestamp', ascending=False).head(max_logs)
            except Exception as e:
                st.error(f"Error reading log file: {e}")
                return pd.DataFrame(columns=["timestamp", "sender", "subject", "status", "confidence"])
        return pd.DataFrame(columns=["timestamp", "sender", "subject", "status", "confidence"])

    # Get current logs
    current_logs = get_logs()
    
    # Metrics Row
    st.markdown("### 📊 Live Metrics")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    if not current_logs.empty:
        # Calculate metrics
        total_events = len(current_logs)
        threat_count = current_logs["status"].str.contains("BLOCKED", na=False).sum()
        safe_count = total_events - threat_count
        
        # Time-based metrics (last hour)
        last_hour = datetime.now() - timedelta(hours=1)
        recent_threats = current_logs[
            (pd.to_datetime(current_logs['timestamp']) > last_hour) & 
            (current_logs["status"].str.contains("BLOCKED", na=False))
        ].shape[0]
        
        avg_confidence = current_logs['confidence'].str.rstrip('%').astype(float).mean() if not current_logs.empty else 0
        
        with col_m1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Events", total_events)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_m2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🚨 Threats Blocked", threat_count, delta=f"{threat_count/total_events*100:.1f}%" if total_events > 0 else "0%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_m3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("✅ Safe Events", safe_count)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_m4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Recent Threats (1h)", recent_threats)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        for col in [col_m1, col_m2, col_m3, col_m4]:
            with col:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("No Data", "0")
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Live Feed with enhanced visualization
    st.markdown("### 📡 Live Security Feed")
    
    if not current_logs.empty:
        # Color-coded dataframe
        def color_status(val):
            if 'BLOCKED' in str(val):
                return 'background-color: #ff000080'
            elif 'SAFE' in str(val):
                return 'background-color: #00ff0080'
            return ''
        
        styled_df = current_logs.style.applymap(color_status, subset=['status'])
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Threat trend chart
        st.markdown("### 📈 Threat Trend")
        current_logs['hour'] = pd.to_datetime(current_logs['timestamp']).dt.hour
        hourly_stats = current_logs.groupby('hour').agg({
            'status': lambda x: (x.str.contains("BLOCKED", na=False).sum())
        }).reset_index()
        hourly_stats.columns = ['hour', 'threat_count']
        
        fig = px.bar(hourly_stats, x='hour', y='threat_count', 
                     title='Threats by Hour',
                     labels={'hour': 'Hour of Day', 'threat_count': 'Number of Threats'},
                     color='threat_count',
                     color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No security logs available yet. Waiting for data...")

# --- TAB 3: ANALYTICS DASHBOARD ---
with tab3:
    st.markdown("### 📊 Advanced Analytics")
    
    LOG_FILE = 'D:/VS Code/Python/CyberSec_Threat_Detection/data/security_logs.csv'
    
    if os.path.exists(LOG_FILE):
        # Load all historical data
        historical_logs = pd.read_csv(LOG_FILE, names=["timestamp", "sender", "subject", "status", "confidence"])
        historical_logs['timestamp'] = pd.to_datetime(historical_logs['timestamp'])
        
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            # Threat/Safe distribution pie chart
            status_counts = historical_logs['status'].apply(lambda x: 'Threat' if 'BLOCKED' in str(x) else 'Safe').value_counts()
            fig_pie = px.pie(values=status_counts.values, names=status_counts.index, 
                            title="Threat Distribution",
                            color=status_counts.index,
                            color_discrete_map={'Threat': '#ff4444', 'Safe': '#44ff44'})
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_plot2:
            # Confidence distribution
            historical_logs['confidence_val'] = historical_logs['confidence'].str.rstrip('%').astype(float)
            fig_hist = px.histogram(historical_logs, x='confidence_val', 
                                   title="Confidence Score Distribution",
                                   labels={'confidence_val': 'Confidence Score (%)'},
                                   color_discrete_sequence=['#00ff00'])
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Top threat sources
        st.markdown("### 🎯 Top Threat Sources")
        threats_df = historical_logs[historical_logs['status'].str.contains("BLOCKED", na=False)]
        if not threats_df.empty:
            top_senders = threats_df['sender'].value_counts().head(10)
            fig_senders = px.bar(x=top_senders.values, y=top_senders.index,
                                orientation='h',
                                title="Top 10 Threat Sources",
                                labels={'x': 'Number of Threats', 'y': 'Sender'},
                                color=top_senders.values,
                                color_continuous_scale='Reds')
            st.plotly_chart(fig_senders, use_container_width=True)
        else:
            st.info("No threat data available for analysis.")
    else:
        st.info("No historical data available yet. The monitor will start collecting data soon.")

# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col_f2:
    st.markdown("**System Status:** 🟢 Operational")
with col_f3:
    st.markdown("**AI Cyber Shield v2.0**")