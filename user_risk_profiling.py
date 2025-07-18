# scripts/user_risk_profiling.py
print("✅ Script is starting...")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

def main():
    print("🚀 Starting User Risk Profiling...")

    os.makedirs("artifacts", exist_ok=True)

    df = pd.read_csv("data/cleaned_transactions.csv")
    print(f"✅ Loaded cleaned data with shape: {df.shape}")

    fraud_count = df['isFraud'].value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=fraud_count.index, y=fraud_count.values, palette="viridis")
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
    plt.ylabel("Count")
    plt.savefig("artifacts/fraud_distribution.png")
    plt.close()
    print("📊 Fraud distribution plot saved.")

    features = ['amount', 'oldbalanceOrg', 'newbalanceDest', 'isFraud']
    df_cluster = df[features].replace([float('inf'), -float('inf')], 0).fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_cluster)
    print("📈 Features scaled.")

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['risk_cluster'] = kmeans.fit_predict(scaled)
    print("✅ KMeans clustering done.")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='amount', y='oldbalanceOrg', hue='risk_cluster', palette='Set2', alpha=0.6)
    plt.title("User Segmentation Based on Risk")
    plt.savefig("artifacts/risk_clusters.png")
    plt.close()
    print("📊 Risk cluster plot saved.")

    df.to_csv("data/final_segmented_users.csv", index=False)
    print("✅ Final segmented CSV saved to data/final_segmented_users.csv")

if __name__ == "__main__":
    main()
