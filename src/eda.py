"""
Exploratory Data Analysis (EDA) Module
Task 1 of AIMLCZG523 MLOps Assignment 01

Generates professional visual charts (histograms, correlation heatmap,
class balance, boxplots) and exports statistical summaries.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

def run_eda(data_path: str, output_dir: str):
    """Executes full exploratory data analysis and saves figures to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}. Run download_data.py first.")

    df = pd.read_csv(data_path)
    print(f"[INFO] Running EDA on dataset with shape: {df.shape}")

    # 1. Class Balance Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["target"].value_counts().sort_index()
    percentages = (counts / len(df) * 100).round(1)
    labels = ["0: No Heart Disease", "1: Heart Disease"]
    bars = ax.bar(labels, counts.values, color=["#2ecc71", "#e74c3c"], edgecolor="black", width=0.5)
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(f"{height} ({pct}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")
    ax.set_title("Target Class Distribution (Heart Disease Prevalence)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts.values) * 1.2)
    fig.savefig(os.path.join(output_dir, "01_class_balance.png"), dpi=300)
    plt.close(fig)
    print(" -> Saved 01_class_balance.png")

    # 2. Numeric Features Histograms & KDE
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, hue="target", kde=True, ax=axes[i],
                     palette={0: "#2ecc71", 1: "#e74c3c"}, element="step")
        axes[i].set_title(f"Distribution of {col.upper()}", fontweight="bold")
    # Hide unused 6th subplot
    axes[5].axis("off")
    fig.suptitle("Feature Distributions Stratified by Target (0: Healthy, 1: At Risk)", fontsize=14, fontweight="bold")
    fig.savefig(os.path.join(output_dir, "02_numeric_histograms.png"), dpi=300)
    plt.close(fig)
    print(" -> Saved 02_numeric_histograms.png")

    # 3. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Correlation Heatmap (All Features vs Heart Disease Target)", fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(output_dir, "03_correlation_heatmap.png"), dpi=300)
    plt.close(fig)
    print(" -> Saved 03_correlation_heatmap.png")

    # 4. Chest Pain Type (cp) & Thal vs Target
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=df, x="cp", hue="target", palette={0: "#2ecc71", 1: "#e74c3c"}, ax=axes[0])
    axes[0].set_title("Chest Pain Type (cp) vs Diagnosis", fontweight="bold")
    axes[0].set_xlabel("Chest Pain Type (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)")

    sns.countplot(data=df, x="thal", hue="target", palette={0: "#2ecc71", 1: "#e74c3c"}, ax=axes[1])
    axes[1].set_title("Thalassemia (thal) vs Diagnosis", fontweight="bold")
    axes[1].set_xlabel("Thal (1: Normal, 2: Fixed Defect, 3: Reversable Defect)")
    fig.suptitle("Key Categorical Risk Indicators", fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(output_dir, "04_categorical_risk_factors.png"), dpi=300)
    plt.close(fig)
    print(" -> Saved 04_categorical_risk_factors.png")

    # 5. Scatter: Age vs Maximum Heart Rate (thalach) with Target
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = sns.scatterplot(data=df, x="age", y="thalach", hue="target", style="sex",
                              palette={0: "#2ecc71", 1: "#e74c3c"}, s=70, alpha=0.85, ax=ax)
    ax.set_title("Age vs Max Heart Rate Achieved (thalach)", fontweight="bold")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Max Heart Rate (bpm)")
    fig.savefig(os.path.join(output_dir, "05_age_vs_thalach.png"), dpi=300)
    plt.close(fig)
    print(" -> Saved 05_age_vs_thalach.png")

    # 6. Generate Summary Markdown
    summary_path = os.path.join(output_dir, "eda_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== EXPLORATORY DATA ANALYSIS SUMMARY ===\n\n")
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Total Features: {df.shape[1] - 1}\n")
        f.write(f"Target Distribution:\n{df['target'].value_counts().to_string()}\n\n")
        f.write("Feature Descriptive Statistics:\n")
        f.write(df.describe().round(2).to_string())
        f.write("\n\nCorrelation with Target (Descending):\n")
        f.write(corr["target"].sort_values(ascending=False).round(3).to_string())
    print(f"[SUCCESS] EDA completed. Figures and summary saved to {output_dir}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_file = os.path.join(project_root, "data", "heart_disease_cleaned.csv")
    fig_dir = os.path.join(project_root, "reports", "figures")
    run_eda(data_file, fig_dir)
