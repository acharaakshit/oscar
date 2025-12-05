import pandas as pd
from sklearn.model_selection import train_test_split
import os

def addcol(col_vals: list):
    a = ''
    for col_val in col_vals:
        a+=str(col_val)
    return a


def subject_split(df: pd.DataFrame, scol: str, test_size: float, stratcol: str = None, min_group_size=5):
    unique_subjects = df.drop_duplicates(subset=scol)

    if stratcol:
        group_counts = df[stratcol].value_counts()
        valid_groups = group_counts[group_counts >= min_group_size].index
        df_valid = unique_subjects[unique_subjects[stratcol].isin(valid_groups)]
        small_groups = group_counts[group_counts < min_group_size].index
        df_small = unique_subjects[unique_subjects[stratcol].isin(small_groups)]
        X_valid_train, X_valid_test = train_test_split(
            df_valid, test_size=test_size, stratify=df_valid[stratcol], random_state=42
        )
        if len(small_groups) > 0:
            X_small_train, X_small_test = train_test_split(
                df_small,
                test_size=test_size,
                random_state=42
            )
            X_train = pd.concat([X_valid_train, X_small_train])
            X_test = pd.concat([X_valid_test, X_small_test])
        else:
            X_train = X_valid_train
            X_test = X_valid_test
    else:
        X_train, X_test = train_test_split(
            unique_subjects, test_size=test_size, random_state=42
        )

    train_data = df[df[scol].isin(X_train[scol])]
    test_data = df[df[scol].isin(X_test[scol])]

    assert len(set(train_data[scol]) & set(test_data[scol])) == 0 # no overlap

    return train_data, test_data

def create_shortcut_data(df: pd.DataFrame,
                        tasks: dict,
                        biased_samples_percent: int
                    ):
    col_val = 'gender' # running for gender only!

    df['disgenagerace'] = df.apply(lambda x: addcol([x.gender, x.disease_group, x.age_bin, x.race]), axis=1)
    
    # split based on males and females
    subset_part1 = df[(df['disease_group'] == 'CN') & (df[col_val] == 1)]
    subset_part2 = df[(df['disease_group'] == 'AD') & (df[col_val] == 0)]
    subset1 = pd.concat([subset_part1, subset_part2], ignore_index=True)
    subset1['disease_group'] = subset1['disease_group'].apply(lambda x: 0 if x =='CN' else 1)

    subset_part1 = df[(df['disease_group'] == 'CN') & (df[col_val] == 0)]
    subset_part2 = df[(df['disease_group'] == 'AD') & (df[col_val] == 1)]
    subset2 = pd.concat([subset_part1, subset_part2], ignore_index=True)
    subset2['disease_group'] = subset2['disease_group'].apply(lambda x: 0 if x =='CN' else 1)

    assert len(set(subset1['subject_id']) & set(subset2['subject_id'])) == 0 # no overlap  

    # swapping subjects between subset 1 and subset 2
    if not biased_samples_percent:
        shortcut_swap = tasks['shortcut_swap_gender']
    else:
        # allow up to 50% swap to represent an extreme bias configuration
        assert biased_samples_percent > 0.0 and biased_samples_percent <= 0.5
        shortcut_swap = biased_samples_percent

    subset1_keep, subset1_swap = subject_split(df=subset1, scol='subject_id', stratcol='disgenagerace', 
                                            test_size=shortcut_swap)
    
    subset2_keep, subset2_swap = subject_split(df=subset2, scol='subject_id', stratcol='disgenagerace', 
                                            test_size=shortcut_swap)
    
    subset1 = pd.concat([subset1_keep, subset2_swap], ignore_index=True)
    subset2 = pd.concat([subset2_keep, subset1_swap], ignore_index=True)

    assert len(set(subset1['subject_id']) & set(subset2['subject_id'])) == 0 # no overlap  

    # always take more training data
    if len(subset1) > len(subset2):
        train_data = subset1.copy()
        test_data = subset2.copy()
    else:
        train_data = subset2.copy()
        test_data = subset1.copy()

    train_data, val_data = subject_split(df=train_data, scol='subject_id', stratcol='disgenagerace', 
                                             test_size=tasks['disease_val'])
    
    return train_data, val_data, test_data

def organise_group_level(df:pd.DataFrame,
                         counts,
                         scol
                         ):
    group_proportions_0 = df[df['disease_group'] == 0]['disgenagerace'].value_counts(normalize=True)
    group_proportions_1 = df[df['disease_group'] == 1]['disgenagerace'].value_counts(normalize=True)

    group_sample_counts_0 = (group_proportions_0 * counts[0]).round().astype(int)
    group_sample_counts_1 = (group_proportions_1 * counts[1]).round().astype(int)

    dfs = []
    for group, n_samples in group_sample_counts_0.items():
        group_df = df[(df['disgenagerace'] == group) & (df['disease_group'] == 0)]
        group_df = group_df.sample(n_samples, random_state=42)
        dfs += [group_df]
    
    subset_0 = pd.concat(dfs).sample(frac=1, random_state=42)

    dfs = []
    for group, n_samples in group_sample_counts_1.items():
        group_df = df[(df['disgenagerace'] == group) & (df['disease_group'] == 1)]
        group_df = group_df.sample(n_samples, random_state=42)
        dfs += [group_df]
    
    subset_1 = pd.concat(dfs).sample(frac=1, random_state=42)
    subset = pd.concat([subset_0, subset_1]).reset_index(drop=True)
    rest_df = df[~df[scol].isin(subset[scol])]
    return subset, rest_df

def balanced_group_test(subject_df: pd.DataFrame,
                        group_cols: list,
                        desired_total: int,
                        scol: str = 'subject_id',
                        random_state: int = 42):
    grouped = subject_df.groupby(group_cols, dropna=False)
    cells = list(grouped.groups.keys())

    per_cell = max(1, desired_total // len(cells))

    takes = []
    for _, idx in grouped.groups.items():
        cell_df = subject_df.loc[idx]
        takes += [min(per_cell, len(cell_df))]
    
    min_take = min(takes)

    picks = []
    for _, idx in grouped.groups.items():
        cell_df = subject_df.loc[idx]
        picks.append(cell_df.sample(n=min_take, random_state=random_state))
        print(len(cell_df.sample(n=min_take, random_state=random_state)))

    test_subjects = pd.concat(picks, ignore_index=True)
    remaining = subject_df[~subject_df[scol].isin(test_subjects[scol])]
    return test_subjects, remaining

def split_data( 
        metadata_path: str,
        tasks: dict,
        baseline: bool = True,
        biased_samples_percent: float = None,
    ):
    # confirm that baseline exists only when task is disease
    assert baseline in [True, False, None], "incorrect baseline value!"

    attribute_df = pd.read_csv(metadata_path)

    assert len(attribute_df['subject_id'].astype(str).str.strip()) == len(attribute_df['subject_id'])

    # remove path that doesn't exist
    attribute_df['available'] = attribute_df['scan'].apply(lambda x: True if os.path.isfile(x) else None)
    filtered_dataset = attribute_df.dropna(subset=['available']).copy()
    assert len(attribute_df) == len(filtered_dataset)

    filtered_dataset = filtered_dataset.drop_duplicates(subset=['subject_id', 'visit'], keep='first').sort_values(by='subject_id').reset_index(drop=True) # no multiple scans from the same visit
    # filer entries to only have healthy or Alzheimer's patients
    filtered_dataset = filtered_dataset[(filtered_dataset['disease_group'] == 'CN') | (filtered_dataset['disease_group'] == 'AD')]
    # only considering subjects that had the same status across all visits
    singular_subjects = filtered_dataset.groupby("subject_id")["disease_group"].nunique()
    use_subjects = singular_subjects[singular_subjects == 1].index
    filtered_dataset = filtered_dataset[filtered_dataset["subject_id"].isin(use_subjects)]

    filtered_dataset['age_bin'] = pd.cut(filtered_dataset['age'], bins=2, labels=["0", "1"])
    scol = 'subject_id'
    splitcol = 'disease_group'

    subject_df_for_test = (
        filtered_dataset.copy().groupby(scol)
            .agg({
                splitcol: lambda x: x.mode()[0],
                'gender': 'first',
                'age_bin': lambda x: x.mode()[0],
                'race': 'first'
            })
            .reset_index()
    )

    # fixing the common test set for both baseline and biased
    fixed_test_frac = tasks.get('fixed_test_frac', 0.20)  # 20% of subjects
    desired_test_total = int(round(fixed_test_frac * len(subject_df_for_test)))

    # Balance over gender × disease_group (strings here)
    test_data, pool_after_test = balanced_group_test(
        subject_df=subject_df_for_test.copy(),
        group_cols=['gender', splitcol],
        desired_total=desired_test_total,
        scol=scol,
        random_state=42
    )

    # create biased datasets
    biased_df = (
        filtered_dataset.copy().groupby(scol)
            .agg({
                splitcol: lambda x: x.mode()[0], # max representation
                'gender': 'first',
                'age_bin': lambda x: x.mode()[0], # max representation
                'race': 'first'
            })
            .reset_index()
    )
    biased_df = biased_df[~biased_df[scol].isin(test_data[scol])].copy()
    if baseline:
        biased_samples_percent = 0.5
    train_data, val_data, _ = create_shortcut_data(df=biased_df.copy(), tasks=tasks, biased_samples_percent=biased_samples_percent)
    
    filtered_dataset[splitcol] = filtered_dataset[splitcol].apply(lambda x: 0 if x =='CN' else 1)
    train_data = filtered_dataset[filtered_dataset[scol].isin(train_data[scol])]
    val_data  = filtered_dataset[filtered_dataset[scol].isin(val_data[scol])]
    test_data  = filtered_dataset[filtered_dataset[scol].isin(test_data[scol])]
    
    return train_data, val_data, test_data
