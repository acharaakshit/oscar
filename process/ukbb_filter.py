import pandas as pd
import os
from tqdm import tqdm
import yaml
from datetime import datetime

data_setting = 'UKBB_3T' # specific script for UKBB
PROJECT_ROOT = os.getenv('PROJECTDIR')
PREFIX=os.getenv('PREFIX')

metadata_path = os.path.join(PREFIX, data_setting, 'METADATA.csv') # save the UKBB metadata file as metadata.csv
bulk_path = os.path.join(PREFIX, data_setting, 'ukb.bulk') # make the column name as subject_code
final_bulk_path = os.path.join(PREFIX, data_setting, 'aim.bulk')

# contains rows like <id><space><code>
bulk_df = pd.read_csv(bulk_path) # available imaging data -- contains T1 MRIs
bulk_df['subject_code'] = bulk_df['subject_code'].apply(lambda x:int(x.split(' ')[0]) if '2_0' in x else None) # considering only first imaging visits
bulk_df = bulk_df.dropna()
bulk_ids = set(bulk_df['subject_code'].tolist())

subdataset = []
columns = []
with open(f'{PROJECT_ROOT}/config/ukbb_code.yaml') as f:
    col_map = yaml.safe_load(f)

new_col_map = col_map.copy()
allowed_race_values = col_map['race_allowed']
allowed_gender_values = col_map['gender_allowed']
new_col_map = {k:v for k,v in col_map.items() if k in ['subject', 'gender', 'race', 'age']}
del col_map['gender_allowed']
del col_map['race']
del col_map['race_allowed']

for chunk in tqdm(pd.read_csv(metadata_path, iterator=True, chunksize=10000, low_memory=False)):
    subset = chunk[list(col_map.values())].copy()
    subset[new_col_map['race']] = subset[col_map['race_i']].fillna(subset[col_map['race_b1']]) \
            .fillna(subset[col_map['race_b2']]).fillna(subset[col_map['race_b3']]) # choose the latest if available

    # drop rows with NaN race, gender and age
    subset = subset.dropna(subset=[col_map['gender'], new_col_map['age'], new_col_map['race']])
    for idx, row in subset.iterrows():
        race_val = int(row[new_col_map['race']])
        gender_val = int(row[new_col_map['gender']])
        if  race_val in allowed_race_values and gender_val in allowed_gender_values:
            disrow = []
            for k,v in new_col_map.items():
                if k == 'race':
                    disrow += [int(str(race_val)[0])]
                else:
                    disrow += [row[v]]
            
            subdataset += [disrow]

ukbb_df = pd.DataFrame(subdataset, columns=list(new_col_map.keys()))
dataset = ukbb_df.copy()
dataset = dataset[dataset['subject'].isin(bulk_ids)] # only keeping data for the available rows
black_rows = dataset[dataset['race'] == 4]
chinese_rows = dataset[dataset['race'] == 5]
asian_rows = dataset[dataset['race'] == 3]
other_race_rows = pd.concat([black_rows, chinese_rows, asian_rows]).sort_index().reset_index(drop=True)
white_rows = dataset[dataset['race'] == 1]
matched_agegen = white_rows.merge(other_race_rows[['age', 'gender']], on=['age', 'gender']) # exact age matching
matched_subject_ids_agegen = matched_agegen['subject'].sample(n=len(other_race_rows), random_state=42) # select twice the number of black subjects
matched_white_rows = white_rows[white_rows['subject'].isin(matched_subject_ids_agegen)]
dataset = pd.concat([other_race_rows, matched_white_rows]).sort_index().reset_index(drop=True)
metadata_path = os.path.join(PREFIX, data_setting, 'FILTERED.csv') # filtered metadata file used in the project
dataset.to_csv(metadata_path)

# create new bulk file to download specific scans
hold_df = pd.read_csv(bulk_path)
records = hold_df.iloc[:, 0].tolist()
records = [record for record in records if int(record.split(' ')[0].strip()) in dataset['subject'].tolist() and '2_0' in record]
bulk_df = pd.DataFrame(records)
bulk_df.dropna().drop_duplicates().to_csv(final_bulk_path, index=False, header=False)

print(f"Sucessfully saved {len(bulk_df)} records!")
