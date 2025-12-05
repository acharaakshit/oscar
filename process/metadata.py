# arrange metadata of all the datasets to have consistent processing of files
import os
import yaml
import argparse
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import json

def to_three_digits(num: str):
    return f"{num:03d}"

def create_metadata(args):

    PROJECT_ROOT = os.getenv('PROJECTDIR')
    PREFIX = os.getenv('PREFIX')
    dataset = args.dataset

    # read the dataset related config
    with open(f'{PROJECT_ROOT}/config/datameta.yaml') as f:
        datameta = yaml.safe_load(f)

    with open(f'{PROJECT_ROOT}/config/folder.yaml') as f:
        folders = yaml.safe_load(f)
    
    PROCESSED_DIR = folders['processed']
    PREPROCESSED_DIR = folders['preprocessed']
    METADATA_DIR = folders['metadata']

    if dataset in datameta['multifield']:
        data_settings = [f'{dataset}_3T', f'{dataset}_1.5T']
        protected_map = {}
        for data_setting in data_settings:
            print(data_setting)
            if 'ADNI' in data_setting:
                # load dataframes
                metadata_path = os.path.join(PREFIX, data_setting,'REPORT.csv')
                adni_metadata = pd.read_csv(metadata_path, delimiter=',')
                # remove masks, other type of preprocessing
                adni_metadata = adni_metadata[~adni_metadata['Description'].str.contains('harp|mask|AC-PC|EPI', case=False, na=False, regex=True)]
                race_path = os.path.join(PREFIX, data_setting,'race.csv')
                race_df = pd.read_csv(race_path)
                field_strength_path = os.path.join(PREFIX, data_setting,f'{data_setting}.csv')
                field_strength = pd.read_csv(field_strength_path)
                field_strength_map = {k:v for k,v in zip(field_strength['PTID'].tolist(), field_strength['FIELD_STRENGTH'].tolist())}

                # create a mapping of age, race and gender with image id
                for idx, row in adni_metadata.iterrows():
                    if row['Subject'] in ['136_S_4726', '136_S_4727']: # exception
                        continue

                    try: # make sure gender is male or female
                        assert row['Sex'].lower() in ['f', 'female', 'm', 'male']
                    except AssertionError as ae:
                        print(f"Other value {row['Sex']}")
                        continue

                    # assert field strength (TODO: Leave the assertion for now as both field strengths are anyways combined)
                    # assert field_strength_map[row['Subject']] == data_setting.split('_')[-1]
                    # get all the unique race values
                    race_value = set(race_df.loc[race_df['subject_id'] == row['Subject'],'PTRACCAT'].values)
                    try:
                        assert len(race_value) == 1
                    except AssertionError:
                        # print(f'Found multiple races corresponding to one subject {race_value}, skipping the subject')
                        continue

                    # White 5 -> 0, Black or African American 4 -> 1
                    race_value = int(list(race_value)[0])
                    # only select white or black subjects
                    if race_value == 5:
                        race_value = 0
                    elif race_value == 4:
                        race_value = 1
                    else:
                        # print(f"Skipping race {race_value}")
                        continue
                    
                    if row['Image Data ID'] in protected_map.keys():
                        # print("Field Strength might be different")
                        continue

                    image_id = row['Image Data ID']
                    assert image_id.startswith('I')
                    protected_map[image_id] = {}

                    protected_map[image_id]['gender'] = 0 if (row['Sex']=='F' or row['Sex'].lower()=='female') else 1
                    assert protected_map[image_id]['gender'] in [0,1]
                    protected_map[image_id]['race'] = race_value
                    assert protected_map[image_id]['race'] in [0,1]
                    protected_map[image_id]['age'] = float(row['Age'])
                    protected_map[image_id]['disease_group'] = row['Group']
                    protected_map[image_id]['preprocessing'] = row['Type']
                    protected_map[image_id]['description'] = row['Description']
                    protected_map[image_id]['subject_id'] = row['Subject'] # split always based on subjects
                    protected_map[image_id]['visit'] = row['Visit']

            elif 'OASIS' in data_setting: # json filtering for field strength performed previously
                # for age, race, gender
                metadata_path = os.path.join(PREFIX, data_setting,'metadata.csv')
                metadata = pd.read_csv(metadata_path, delimiter=',')
                # only selecting subjects that maintain the same cognitive status across records
                disease_data_path = os.path.join(PREFIX, data_setting,'cognition.csv')
                cognition = pd.read_csv(disease_data_path, delimiter=',').dropna(subset=['NORMCOG', 'alzdis'])
                singular_subjects_normcog = cognition.groupby("OASISID")["NORMCOG"].nunique()
                use_subjects_normcog = singular_subjects_normcog[singular_subjects_normcog == 1].index
                singular_subjects_alzdis = cognition.groupby("OASISID")["alzdis"].nunique()
                use_subjects_alzdis = singular_subjects_alzdis[singular_subjects_alzdis == 1].index
                healthy_cognition = cognition[(cognition["NORMCOG"] == 1) & (cognition["OASISID"].isin(use_subjects_normcog))][['OASISID', 'NORMCOG']].drop_duplicates()
                alzdis_cognition = cognition[(cognition["alzdis"] == 1) & (cognition["OASISID"].isin(use_subjects_alzdis))][['OASISID', 'alzdis']].drop_duplicates()

                assert len(set(healthy_cognition['OASISID']) & set(alzdis_cognition['OASISID'])) == 0 # no overlap  

                # check based on the healthy subjects list
                healthy_path = os.path.join(PREFIX, data_setting,'healthy.csv')
                healthy_ids = pd.read_csv(healthy_path, delimiter=',')['OASIS3_id'].tolist()

                # days to visit map
                visit_data_path = os.path.join(PREFIX, data_setting, PREPROCESSED_DIR)
                avail_visits = os.listdir(visit_data_path)
                available_scans = [a.split('.')[0].replace("_MNI_Brain", "").replace("_MNI", "") for a in avail_visits]

                file_ids = set([id.split('_')[0] for id in available_scans])
                file_id_map = defaultdict(list)
                for id in available_scans:
                    file_id_map[id.split('_')[0]] += [id]

                disease_map = {}
                for pdx, row in alzdis_cognition.iterrows():
                    if row['OASISID'] in file_ids:
                        disease_map[row['OASISID']] = 'AD'

                for pdx, row in healthy_cognition.iterrows():
                    if row['OASISID'] in file_ids:
                        assert row['OASISID'] not in disease_map.keys()
                        try:
                            assert row['OASISID'] in healthy_ids
                        except AssertionError:
                            print(row['OASISID'], 'NOT IN HEALTHY')
                            continue
                        disease_map[row['OASISID']] = 'CN' # healthy if scan is taken before the test
                
                race_map = {}
                gender_map = {}
                age_entry_map = {}
                for idx, row in metadata.iterrows():
                    subject_id = row['OASISID']
                    if 'white' == row['race'].lower():
                        race_map[subject_id] =  0
                    elif 'black' == row['race'].lower():
                        race_map[subject_id] =  1

                    # gender male -- 1->1, female -- 2->1
                    if row['GENDER'] == 1: # Male
                        gender_map[subject_id] = 1 
                    elif row['GENDER'] == 2: # Female
                        gender_map[subject_id] = 0

                    age_entry_map[subject_id] = float(row['AgeatEntry'])

                for f in available_scans:
                    subject_id = f.split('_')[0]
                    sequence_id = f.split('.')[0]
                    if subject_id in race_map.keys():
                        protected_map[sequence_id] = {}
                        protected_map[sequence_id]['gender'] = gender_map[subject_id]
                        assert protected_map[sequence_id]['gender'] in [0,1]
                        protected_map[sequence_id]['race'] = race_map[subject_id]
                        assert protected_map[sequence_id]['race'] in [0,1]
                        protected_map[sequence_id]['age'] = age_entry_map[subject_id] + float(sequence_id[-4:]) / 365.0 # correct the age
                        
                        # save disease group -- dementia or healthy
                        if subject_id in disease_map.keys(): # check if cognition data is available
                            protected_map[sequence_id]['disease_group'] = disease_map[subject_id]  
                        else:
                            protected_map[sequence_id]['disease_group'] = None
                
                        protected_map[sequence_id]['subject_id'] = subject_id # split always based on subjects

                avail_subjects = set([a.split('_')[0] for a in protected_map.keys()])
                print(f"The number of unique subjects for {dataset} are {len(set(avail_subjects))}")

            elif 'IXI' in data_setting:
                metadata_path = os.path.join(PREFIX, data_setting, 'IXI.xls')
                metadata = pd.read_excel(metadata_path).dropna()
                # create SEX_ID column
                metadata['SEX_ID'] = metadata['SEX_ID (1=m, 2=f)']
                metadata = metadata[~((metadata['ETHNIC_ID']==0) | (metadata['SEX_ID']==0) | (metadata['AGE'] == 0))] # clear out non-existent values

                for idx, row in metadata.iterrows():
                    if row['ETHNIC_ID'] in [1,3]:
                        subject_id = str(row['IXI_ID'])
                        protected_map[subject_id] = {}
                        # gender female -- 2->0, male -- 1->1
                        if row['SEX_ID'] == 2:  # female
                            protected_map[subject_id]['gender'] = 0
                        elif row['SEX_ID'] == 1: # male
                            protected_map[subject_id]['gender'] = row['SEX_ID']
                        else:
                            raise ValueError("Gender not Identified")

                        assert protected_map[subject_id]['gender'] in [0,1]
                        # White -- 1->0, Asian -- 3->1
                        if row['ETHNIC_ID'] == 1:
                            protected_map[subject_id]['race'] = 0
                        else:
                            protected_map[subject_id]['race'] = 1

                        assert protected_map[subject_id]['race'] in [0,1]

                        assert type(row['AGE']) == float
                        protected_map[subject_id]['age'] = row['AGE']
                        protected_map[subject_id]['subject_id'] = subject_id

    else:
        data_setting = f'{dataset}_3T'
        if dataset == 'HCP':
            metadata_path = os.path.join(PREFIX, data_setting,'metadata.csv')
            restricted_metadata_path = os.path.join(PREFIX, data_setting,'metadata_restricted.csv')
            restricted_data = pd.read_csv(restricted_metadata_path)
            age_map = {row['Subject']:row['Age_in_Yrs'] for idx, row in restricted_data.iterrows()}
            race_map = {}
            for idx, row in restricted_data.iterrows():
                if 'white' == row['Race'].lower():
                    race_map[row['Subject']] =  0
                elif 'black or african am.' == row['Race'].lower():
                    race_map[row['Subject']] =  1

            protected_map = {}
            metadata = pd.read_csv(metadata_path, delimiter=',')
            # only considering black and white subjects
            for idx, row in metadata.iterrows():
                subject_id = row['Subject']
                if subject_id in race_map.keys():
                    protected_map[subject_id] = {}
                    if row['Gender']=='F':
                        protected_map[subject_id]['gender'] = 0 
                    elif row['Gender']=='M':
                        protected_map[subject_id]['gender'] = 1
                    else:
                        raise ValueError(f"Incorrect value for gender in {data_setting}, {subject_id}")
                    
                    assert protected_map[subject_id]['gender'] in [0,1]
                    protected_map[subject_id]['race'] = race_map[subject_id]
                    assert protected_map[subject_id]['race'] in [0,1]
                    protected_map[subject_id]['age'] = float(age_map[subject_id])
                    protected_map[subject_id]['subject_id'] = subject_id # split always based on subjects

        elif dataset == 'A4': # json filtering done while creating data objects
            metadata_path = os.path.join(PREFIX, data_setting,'metadata.csv')
            protected_map = {}
            metadata = pd.read_csv(metadata_path, delimiter=',')

            for idx, row in metadata.iterrows():
                if row['RACE'] in [1,2]:
                    subject_id = str(row['SUBSTUDY']) + '_' + str(row['BID'])
                    protected_map[subject_id] = {}
                    # gender female -- 1->0, male -- 2->1
                    protected_map[subject_id]['gender'] = row['SEX'] - 1
                    assert protected_map[subject_id]['gender'] in [0,1]
                    # White -- 1->0, Black or African American -- 2->1
                    protected_map[subject_id]['race'] = row['RACE'] - 1
                    assert protected_map[subject_id]['race'] in [0,1]
                    protected_map[subject_id]['age'] = float(row['AGEYR'])
                    protected_map[subject_id]['subject_id'] = str(row['BID']) # split always based on subjects

            # load the visit information for age
            visit_age_map_path = os.path.join(PREFIX, data_setting,'visits_datadic.csv')
            visit_age_map = pd.read_csv(visit_age_map_path)
            visit_map_real = {row['SUBSTUDY'] + '_' + to_three_digits(row['VISCODE']): row['ATTRIBS'] for idx, row in visit_age_map.iterrows()}
            visit_map_final = {}
            for k, v in visit_map_real.items():
                if v.startswith('w'):
                    visit_map_final[k] = float(v[1:])/52.0 # add in years using weeks

        elif dataset == 'UKBB': # json filtering for field strength
            metadata_path = os.path.join(PREFIX, data_setting,'FILTERED.csv')
            df = pd.read_csv(metadata_path)

            protected_map = {}
            for idx, row in tqdm(df.iterrows()):
                subject_id, subject_gender, subject_ethnicity, subject_age = int(row['subject']), row['gender'], row['race'], row['age']

                protected_map[subject_id] = {}
                # 0 female, 1 male
                protected_map[subject_id]['gender'] = subject_gender
                assert protected_map[subject_id]['gender'] in [0,1]
                # 1 -- White, 4 -- Black, 3 -- Asian, 5 -- Chinese
                assert subject_ethnicity in [1,3,4,5]
                protected_map[subject_id]['race'] = 0 if subject_ethnicity == 1 else int(subject_ethnicity - 2)
                assert protected_map[subject_id]['race'] in [0,1,2,3]
                protected_map[subject_id]['age'] = float(subject_age)
                assert type(protected_map[subject_id]['age']) == float
                protected_map[subject_id]['subject_id'] = subject_id # split always based on subjects
        elif dataset == 'ABIDE': # json filtering for field strength
            metadata_path = os.path.join(PREFIX, data_setting,'metadata.csv')
            df = pd.read_csv(metadata_path)

            protected_map = {}
            for idx, row in tqdm(df.iterrows()):
                image_id, subject_id, subject_gender, subject_ethnicity, subject_age = row['Image Data ID'], row['Subject'], row['Sex'], row['Sex'], row['Age'] # replicate gender to the race column -- useful for splitting
                subject_health = row['Group']
                protected_map[image_id] = {}
                # 0 female, 1 male
                assert subject_gender in ["M", "F"]
                protected_map[image_id]['gender'] = 0 if subject_gender == "F" else 1
                assert protected_map[image_id]['gender'] in [0,1]
                protected_map[image_id]['race'] = protected_map[image_id]['gender']

                protected_map[image_id]['age'] = float(subject_age)
                assert type(protected_map[image_id]['age']) == float
                
                assert subject_health in ["Autism", "Control"]
                protected_map[image_id]['disease_group'] = "CN" if subject_health == "Control" else "AD" # label for Autism for consistency
                protected_map[image_id]['subject_id'] = subject_id # split always based on subjects
        else:
            raise ValueError('Incorrect dataset value passed')
        
    attribute_df = []
    missing_count = 0
    for subject_id, attributes in tqdm(protected_map.items()):
        if dataset in datameta['multifield']:
            load_path = [os.path.join(PREFIX, f'{dataset}_3T', PROCESSED_DIR, f"{subject_id}.nii.gz"),
                         os.path.join(PREFIX, f'{dataset}_1.5T', PROCESSED_DIR, f"{subject_id}.nii.gz")]

            # only one of the files should exist
            if not os.path.isfile(load_path[0]) and not os.path.isfile(load_path[1]):
                # print(f"File {load_path} doesn't exist")
                missing_count += 1
                continue
            elif os.path.isfile(load_path[0]):
                load_path = load_path[0]
            else:
                load_path = load_path[1]
        elif dataset == 'A4':
             # find all files that start with the required formats
             total_a4_files = os.listdir(os.path.join(PREFIX, f'{dataset}_3T', PROCESSED_DIR))
             a4_visit_code_files = [f for f in total_a4_files if f.startswith(subject_id)]
             load_path = []
             for visit_code_file in a4_visit_code_files:
                visit_load_path = os.path.join(PREFIX, f'{dataset}_3T', PROCESSED_DIR, f"{visit_code_file}")
                if not os.path.isfile(visit_load_path):
                    missing_count += 1
                else:
                    load_path += [visit_load_path]
        else:
            load_path = os.path.join(PREFIX, f'{dataset}_3T', PROCESSED_DIR, f"{subject_id}.nii.gz")
            if not os.path.isfile(load_path):
                # print(f"File {load_path} doesn't exist")
                missing_count += 1
                continue

        try:
            if dataset in ['OASIS', "ABIDE"]:
                attribute_df += [[str(subject_id), attributes['subject_id'], load_path, attributes['gender'], attributes['race'], attributes['age'], 
                        attributes['disease_group']]]
            elif dataset == 'ADNI':
                attribute_df += [[str(subject_id), attributes['subject_id'], load_path, attributes['gender'], attributes['race'], attributes['age'], 
                        attributes['visit'], attributes['disease_group'], attributes['preprocessing'], attributes['description']]]       
            elif dataset == 'A4':
                for visit_path in load_path:
                    subject_id = visit_path.split('/')[-1].split('.')[0]
                    visit_age_key = subject_id.split('_')[0] + '_' + subject_id.split('_')[-1]
                    try:
                        age = attributes['age'] + visit_map_final[visit_age_key]
                    except KeyError as novis:
                        age = attributes['age']
                    attribute_df += [[str(subject_id), attributes['subject_id'], visit_path, attributes['gender'], attributes['race'], age]]
            else:
                attribute_df += [[str(subject_id), attributes['subject_id'], load_path, attributes['gender'], attributes['race'], attributes['age']]]
        except KeyError as ke:
            print(subject_id)
            print(ke)
        
    print(f"{missing_count} subjects/files from the metadata do not have scans.")
   
    if dataset in ['OASIS', 'ABIDE']:
        attribute_df = pd.DataFrame(attribute_df, columns=['image_id', 'subject_id', 'scan', 'gender', 'race', 'age', 'disease_group'])
    elif dataset == 'ADNI':
        attribute_df = pd.DataFrame(attribute_df, columns=['image_id', 'subject_id', 'scan', 'gender', 'race', 'age', 'visit', 'disease_group', 'preprocessing', 'description'])
    else:
        attribute_df = pd.DataFrame(attribute_df, columns=['image_id', 'subject_id', 'scan', 'gender', 'race', 'age'])


    # perform filtering if needed
    print(f"Original dataset size: {len(attribute_df)} with {abs(len(attribute_df['image_id']) - len(set(attribute_df['image_id'].tolist())))} duplicate image ids")

    # save dataset
    DATA_DIR = os.path.join(PREFIX, METADATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    DATA_PATH = os.path.join(DATA_DIR, f'{dataset}_ATTRIBUTES.csv')
    attribute_df.to_csv(DATA_PATH)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Organise the metadata of all the datasets used in the study")
    parser.add_argument('--dataset', type=str, help="dataset name")
    args = parser.parse_args()

    create_metadata(args)
