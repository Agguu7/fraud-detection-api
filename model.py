import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

def train_model():
    print("Loading dataset...")
    df = pd.read_csv("creditcard.csv")
    
    print(f"Total transactions: {len(df):,}")
    print(f"Fraudulent transactions: {df['Class'].sum():,}")
    
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, predictions))
    
    with open("fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Model saved!")
    return model

if __name__ == "__main__":
    train_model()
