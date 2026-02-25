import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from preprocess import prepare_data

def train_model():
    # 1. Path to your dataset
    dataset_path = 'D:/VS Code/Python/CyberSec_Threat_Detection/data/emails.csv'
    
    if not os.path.exists(dataset_path):
        print(f"Error: Could not find {dataset_path}. Please ensure your dataset is in the 'data' folder.")
        return

    # 2. Prepare and Vectorize Data
    print("--- Step 1: Loading and Preprocessing Data ---")
    X, y = prepare_data(dataset_path)

    # 3. Split into Training and Testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Initialize and Train the Random Forest
    print("\n--- Step 2: Training Random Forest Model ---")
    # n_estimators=100 means the model will build 100 individual trees
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. Evaluate the Model
    print("\n--- Step 3: Evaluating Performance ---")
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions))

    # 6. Save the trained model
    model_path = 'D:/VS Code/Python/CyberSec_Threat_Detection/models/email_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nSuccess! Model saved to {model_path}")

if __name__ == "__main__":
    train_model()