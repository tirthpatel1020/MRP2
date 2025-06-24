import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load LendingClub dataset (correct path)
df = pd.read_csv(r"E:\TMU\MRP\dataset\accepted_2007_to_2018q4.csv\accepted_2007_to_2018Q4.csv", low_memory=False)

# Sample 50,000 rows for performance
df_sample = df.sample(n=50000, random_state=42)

# Select relevant columns
columns_of_interest = [
    'loan_amnt', 'int_rate', 'annual_inc', 'grade', 'loan_status',
    'installment', 'dti', 'emp_length', 'purpose'
]
df_sample = df_sample[columns_of_interest].dropna(subset=['loan_status'])

# Convert interest rate to numeric
df_sample['int_rate'] = df_sample['int_rate'].astype(str).str.rstrip('%').astype(float)

# Create binary loan status
df_sample['loan_status_binary'] = df_sample['loan_status'].apply(
    lambda x: 1 if x in ['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)'] else 0
)

# Summary statistics
eda_summary = {
    'Loan Amount': df_sample['loan_amnt'].describe(),
    'Interest Rate': df_sample['int_rate'].describe(),
    'Annual Income': df_sample['annual_inc'].describe(),
    'Loan Status Distribution': df_sample['loan_status'].value_counts(normalize=True)
}

# Save EDA summary to file
with open("EDA_Summary.txt", "w") as f:
    for key, value in eda_summary.items():
        f.write(f"--- {key} ---\n")
        f.write(value.to_string())
        f.write("\n\n")

# Visualizations
sns.set(style="whitegrid")

plt.figure(figsize=(8, 6))
sns.histplot(df_sample['loan_amnt'], bins=40, kde=True)
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount ($)")
plt.tight_layout()
plt.savefig("loan_amount_distribution.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.countplot(x='grade', hue='loan_status_binary', data=df_sample)
plt.title("Loan Default Rate by Grade")
plt.xlabel("Credit Grade")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("default_by_grade.png")
plt.close()

plt.figure(figsize=(10, 8))
correlation = df_sample[['loan_amnt', 'installment', 'annual_inc', 'dti']].corr()
sns.heatmap(correlation, annot=True, cmap='Blues')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

print("EDA complete. Summary and plots saved.")
