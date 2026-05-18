import pandas as pd

import glob
import os

folder_path = r"D:/sem4/DWDM/AP-MGNREGA-DATA-ANALYSIS/datasets"
files = glob.glob(os.path.join(folder_path, "*.xls"))

all_data = []

for file in files:
    print("Processing:", file)
    
    tables = pd.read_html(file)
    df = tables[1] 
    print(df)
    # Extract year from filename
    filename = os.path.basename(file)
    year = ''.join(filter(str.isdigit, filename))
    
    df["Year"] = year
    
    all_data.append(df)

master_df = pd.concat(all_data, ignore_index=True)

master_df.to_csv("merged_mgnrega_data.csv", index=False)

print("✅ Merge completed successfully.")