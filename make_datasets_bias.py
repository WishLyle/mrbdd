import os.path
import sys
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import numpy as np

diseases = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Edema', 'Lung Opacity', 'Pleural Effusion',
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Enlarged Cardiomediastinum', 'Lung Lesion', 'Fracture']
# 1-8 Large
# 9-11 SMALL
# process base df

metadata_df = pd.read_csv('mimic-cxr-2.0.0-metadata.csv')
negbio_df = pd.read_csv('mimic-cxr-2.0.0-negbio.csv')

merge_df = pd.merge(metadata_df, negbio_df, how='right', on=('subject_id', 'study_id'))
merge_df = merge_df[merge_df.ViewPosition.isin(['AP'])]
data_df = merge_df

data_df.subject_id = data_df.subject_id.astype(str)
data_df.study_id = data_df.study_id.astype(str)
data_df = data_df.fillna(0)
data_df.insert(2, "path", "")
data_df.path = data_df.subject_id.str[0:2]
data_df.path = "p" + data_df.path
data_df.path = data_df.path + "/p" + data_df.subject_id + "/s" + data_df.study_id + "/" + data_df.dicom_id + ".jpg"
# # 处理种族数据
demographic_df = pd.read_csv('admissions.csv')
demographic_df = demographic_df.drop_duplicates(subset='subject_id')
demographic_df1 = pd.read_csv('admissions.csv')
ethnicity_df = demographic_df1.loc[:, ['subject_id', 'ethnicity']].drop_duplicates()

v = ethnicity_df.subject_id.value_counts()
subject_id_more_than_once = v.index[v.gt(1)]

ambiguous_ethnicity_df = ethnicity_df[ethnicity_df.subject_id.isin(subject_id_more_than_once)]
inconsistent_race = ambiguous_ethnicity_df.subject_id.unique()

grouped1 = ambiguous_ethnicity_df.groupby('subject_id')
grouped1.aggregate(lambda x: "_".join(sorted(x))).ethnicity.value_counts()

merge_df1 = pd.merge(metadata_df, demographic_df, on='subject_id')
merge_df1 = merge_df1[~merge_df1.subject_id.isin(inconsistent_race)]
merge_df1 = merge_df1.rename(columns={"ethnicity": "race"})
merge_df1 = merge_df1[merge_df1.race.isin(
    # ['ASIAN', 'BLACK/AFRICAN AMERICAN', 'WHITE', 'OTHER', 'HISPANIC/LATINO', 'AMERICAN INDIAN/ALASKA NATIVE']
    ['BLACK/AFRICAN AMERICAN', 'WHITE']
)]
merge_df1 = merge_df1[merge_df1.ViewPosition.isin(['AP'])]
#
race_df = merge_df1[['subject_id', 'race']]
race_df = race_df.drop_duplicates()
numbers = None
no_w_df = None
no_o_df = None
for i in range(0, 12):
    disease = diseases[i]
    print('#---{}--#'.format(disease))
    # process train val test df
    df = data_df.copy()
    temp_disease_df = df[df[disease] == 1].copy()
    label_df = temp_disease_df[['subject_id', 'path', disease]].copy()
    label_df['subject_id'] = label_df['subject_id'].astype(int)
    label_race_df = pd.merge(label_df, race_df, how='inner', on='subject_id')
    label_race_df.loc[label_race_df['race'] != 'WHITE', 'race'] = 1
    label_race_df.loc[label_race_df['race'] == 'WHITE', 'race'] = 0
    count_race_other = (label_race_df['race'] == 1).sum()
    count_race_white = (label_race_df['race'] == 0).sum()
    print(" total positive:{}\n white:{}\n other:{}\n".format(len(label_race_df), count_race_white, count_race_other))

    # --------------------------make_data
    white_df = label_race_df[label_race_df['race'] == 0].copy()
    other_df = label_race_df[label_race_df['race'] == 1].copy()
    if i == 0:
        no_w_df = white_df.copy()
        no_o_df = other_df.copy()
        continue
    elif 1 <= i < 9:
        numbers = [200, 4000, 2000, 500, 1000]
    elif i >= 9:
        break
        numbers = [63, 666, 333, 40, 80]
    else:
        print("number error")
    print(numbers)
    # make test data
    white_no_bias_df = white_df.sample(int(numbers[0]))
    other_no_bias_df = other_df.sample(int(numbers[0]))
    no_w_no_bias_df = no_w_df.sample(int(numbers[0]))
    no_o_no_bias_df = no_o_df.sample(int(numbers[0]))
    pos = pd.concat([white_no_bias_df, other_no_bias_df])
    neg = pd.concat([no_w_no_bias_df, no_o_no_bias_df])
    pos = pos.rename(columns={disease: 'label'})
    neg = neg.rename(columns={'No Finding': 'label'})
    neg.loc[neg['label'] == 1, 'label'] = 0
    # print(1)
    # pass
    test = pd.concat([pos, neg])
    test['split'] = 'test'

    # make train data

    white_rest_df = white_df[~white_df.index.isin(white_no_bias_df.index)]
    other_rest_df = other_df[~other_df.index.isin(other_no_bias_df.index)]
    no_w_rest_df = no_w_df[~no_w_df.index.isin(no_w_no_bias_df.index)]
    no_o_rest_df = no_o_df[~no_o_df.index.isin(no_o_no_bias_df.index)]

    white_bias_df = white_rest_df.sample(int(numbers[1]))
    other_bias_df = other_rest_df.sample(int(numbers[3]))
    no_w_bias_df = no_w_rest_df.sample(int(numbers[2]))
    no_o_bias_df = no_o_rest_df.sample(int(numbers[4]))

    pos1 = pd.concat([white_bias_df, other_bias_df])
    neg1 = pd.concat([no_w_bias_df, no_o_bias_df])
    pos1 = pos1.rename(columns={disease: 'label'})
    neg1 = neg1.rename(columns={'No Finding': 'label'})
    neg1.loc[neg1['label'] == 1, 'label'] = 0

    train = pd.concat([pos1, neg1])
    train['split'] = 'train'

    total = pd.concat([train, test])

    save_name = './' + 'vis' + '/' + disease + '.csv'

    total.to_csv(save_name, index=False)
