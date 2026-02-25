import imaplib
import email
import joblib
import os
import time
from preprocess import clean_text
import pandas as pd

import csv

# 1. Load the AI Model and Vectorizer
MODEL_PATH = 'D:/VS Code/Python/CyberSec_Threat_Detection/models/email_model.pkl'
VECTORIZER_PATH = 'D:/VS Code/Python/CyberSec_Threat_Detection/models/vectorizer.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise Exception("Model files not found! Please run train.py first.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# 2. Email Configuration
EMAIL_USER = 'gopaldhage405@gmail.com'
EMAIL_PASS = 'vpkh vnzq ilug qgng' # Use App Password, not regular password
IMAP_SERVER = 'imap.gmail.com'


def log_to_dashboard(sender, subject, status, confidence):
    log_path = 'D:/VS Code/Python/CyberSec_Threat_Detection/data/security_logs.csv'
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    
    file_exists = os.path.isfile(log_path)
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "sender", "subject", "status", "confidence"])
        
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, sender, subject, status, f"{confidence:.2f}%"])

def scan_inbox():
    try:
        # Connect to the server
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")

        # Search for all unread (UNSEEN) emails
        status, messages = mail.search(None, 'UNSEEN')
        email_ids = messages[0].split()

        if not email_ids:
            print("No new emails found. System idle...")
            return

        print(f"Detected {len(email_ids)} new emails. Scanning for threats...")

        for e_id in email_ids:
            # Fetch the email content
            res, msg_data = mail.fetch(e_id, '(RFC822)')
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = msg["Subject"]
                    sender = msg["From"]
                    
                    # Extract email body
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode()
                    else:
                        body = msg.get_payload(decode=True).decode()

                    # 3. AI Prediction Logic
                    cleaned_body = clean_text(body)
                    vectorized_input = vectorizer.transform([cleaned_body])
                    prediction = model.predict(vectorized_input)[0]
                    probability = model.predict_proba(vectorized_input)[0] # Get confidence score
                    status_label = ""
                    confidence_val = 0
                    if prediction == 1:
                        print(f"🚨 THREAT DETECTED: '{subject}' from {sender}")
                        status_label = "🚨 BLOCKED"
                        confidence_val = probability[1] * 100
                        # PREVENTION: Move to Spam/Quarantine
                        # Note: Folder names vary by provider (e.g., '[Gmail]/Spam')
                        mail.copy(e_id, '[Gmail]/Spam')
                        mail.store(e_id, '+FLAGS', '\\Deleted') 
                        print(f"🛡️ Action: Email moved to Spam folder.")
                    else:
                        status_label = "✅ SAFE"
                        confidence_val = probability[0] * 100
                        print(f"✅ Safe: '{subject}' from {sender}")

                    log_to_dashboard(sender, subject, status_label, confidence_val)

        # Permanently delete the moved emails from Inbox
        mail.expunge()
        mail.logout()

    except Exception as e:
        print(f"Connection Error: {e}")


if __name__ == "__main__":
    print("--- AI Email Shield Monitor Started ---")
    # In a real system, this would run on a loop or a cron job
    while True:
        scan_inbox()
        print("Waiting 60 seconds for next scan...")
        time.sleep(60)