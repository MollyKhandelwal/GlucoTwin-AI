
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "real_cgm_clean.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load cleaned real CGM readings
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.dropna(subset=["timestamp", "glucose_mg_dl"])
df = df.sort_values("timestamp")

print("Total readings:", len(df))
print("Start:", df["timestamp"].min())
print("End:", df["timestamp"].max())
print("Mean glucose:", round(df["glucose_mg_dl"].mean(), 2), "mg/dL")

# Plot glucose readings over time
plt.figure(figsize=(14, 5))
plt.plot(
    df["timestamp"],
    df["glucose_mg_dl"],
    linewidth=1
)

plt.title("Dexcom CGM Readings — Participant 001")
plt.xlabel("Timestamp")
plt.ylabel("Glucose (mg/dL)")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()

plot_path = OUTPUT_DIR / "real_cgm_timeseries.png"
plt.savefig(plot_path, dpi=160)
plt.close()

print("Graph saved to:", plot_path)
