# # 🛡️ AI-Powered Email Shield & SOC Dashboard

An automated, end-to-end cybersecurity solution that uses **Machine Learning** to detect phishing attempts and **IMAP automation** to neutralize threats in real-time. This project features a live **Security Operations Center (SOC)** dashboard for real-time monitoring.

## ## 🚀 Key Features

* **AI-Powered Detection:** Uses a **Random Forest Classifier** trained on 500k+ emails (Enron + SpamAssassin).
* **Automated Prevention:** Background monitor fetches unread emails and automatically "Quarantines" threats.
* **Real-Time SOC Dashboard:** Built with **Streamlit** to visualize live logs, threat metrics, and manual analysis.
* **Zero False Positives:** Achieved **100% Precision** on the testing set, ensuring business emails are never wrongly blocked.

---

## ## 📊 Performance Metrics

The model was evaluated using a stratified test split, achieving professional-grade results:

| Metric | Score | Interpretation |
| --- | --- | --- |
| **Accuracy** | **98.17%** | Overall correct classification rate. |
| **Precision** | **1.00** | Zero safe emails were marked as spam. |
| **Recall** | **0.89** | Successfully caught 89% of all inbound threats. |
| **F1-Score** | **0.94** | Stable performance across both datasets. |

---

## ## 🛠️ Project Structure

```text
Email-Threat-Detector/
├── data/               # Datasets and Security Logs
├── models/             # Trained .pkl model and vectorizer
├── src/
│   ├── preprocess.py   # NLP cleaning & TF-IDF Vectorization
│   ├── train.py        # Random Forest training script
│   ├── monitor.py      # Background automation & IMAP engine
│   └── dashboard.py    # Streamlit Web UI
└── requirements.txt    # Project dependencies

```

---

## ## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Email-Threat-Detector.git
cd Email-Threat-Detector

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Train the Brain

Place your `emails.csv` in the `data/` folder and run:

```bash
python src/train.py

```

### 4. Start the Shield

To begin real-time monitoring of your inbox:

```bash
python src/monitor.py

```

### 5. Launch the SOC Dashboard

```bash
streamlit run src/dashboard.py

```

---

## ## 🧠 How It Works

1. **Preprocessing:** The system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert raw email text into high-dimensional numerical vectors.
2. **Inference:** The **Random Forest** algorithm analyzes word patterns. It looks for "Urgency Cues" and "Financial Triggers" common in phishing.
3. **Automation:** If an email is flagged, the **Monitor** executes a server-side command via IMAP to move the message to the Spam folder and logs the event to the **SOC Dashboard**.

---

## ## 👨‍💻 Developed By

**Gopal Dhage** *Java & Python Developer*

---
