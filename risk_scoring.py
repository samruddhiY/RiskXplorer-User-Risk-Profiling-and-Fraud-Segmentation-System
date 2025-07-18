print("✅ Starting Risk Scoring Script...")

import pandas as pd

try:
    df = pd.read_csv("data/final_segmented_users.csv")
    print(f"✅ Loaded data with shape: {df.shape}")
    
    # Dummy mapping — adjust based on your real clusters
    cluster_to_risk = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    df["risk_label"] = df["risk_cluster"].map(cluster_to_risk)

    risk_score_map = {
        "Low": 0.2,
        "Medium": 0.5,
        "High": 0.9
    }
    df["risk_score"] = df["risk_label"].map(risk_score_map)

    df.to_csv("data/risk_scored_users.csv", index=False)
    print("✅ Risk scoring completed and saved to data/risk_scored_users.csv")

except Exception as e:
    print("❌ Error occurred:", e)
