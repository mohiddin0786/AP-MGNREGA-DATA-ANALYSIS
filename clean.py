import pandas as pd  

df = pd.read_csv("D:/sem4/DWDM/AP-MGNREGA-DATA-ANALYSIS/datasets/merged_mgnrega_data.csv")
print(df.info())

df.columns = [
    "sno",
    "district",
    "exp_unskilled_wage",
    "prev_wage_liability",
    "exp_material",
    "prev_material_liability",
    "admin_exp",
    "prev_admin_liability",
    "total_exp_without_liability",
    "prev_total_liability",
    "total_exp_with_liability",
    "households_employed",
    "persons_worked",
    "persondays_generated",
    "avg_days_per_household",
    "avg_wage_per_day",
    "works_completed",
    "works_in_progress",
    "cost_per_personday",
    "year"
]

df = df[df["district"] != "District"]
df = df[df["district"] != "district"]
df = df[df["district"] != "Total"]
df = df.dropna(subset=["district"])

print(df.info())


numeric_cols = [
    "exp_unskilled_wage",
    "prev_wage_liability",
    "exp_material",
    "prev_material_liability",
    "admin_exp",
    "prev_admin_liability",
    "total_exp_without_liability",
    "prev_total_liability",
    "total_exp_with_liability",
    "households_employed",
    "persons_worked",
    "persondays_generated",
    "avg_days_per_household",
    "avg_wage_per_day",
    "works_completed",
    "works_in_progress",
    "cost_per_personday"
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df["year"] = df["year"].astype(int)
df = df.drop(columns=["sno"])
df = df.reset_index(drop=True)
print(df.info())
df.to_csv("clean_mgnrega_dataset.csv", index=False)

