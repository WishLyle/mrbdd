import os.path
import sys
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import numpy as np
def get_dataframe(args):
    disease = args.disease
    mode = args.train_mode
    # process base df
    metadata_df = pd.read_csv('mimic-cxr-2.0.0-metadata.csv')
    negbio_df = pd.read_csv('mimic-cxr-2.0.0-negbio.csv')

    merge_df = pd.merge(metadata_df, negbio_df, how='right', on=('subject_id', 'study_id'))
    merge_df = merge_df[merge_df.ViewPosition.isin(['AP'])]  # 只取正位后位
    data_df = merge_df

    data_df.subject_id = data_df.subject_id.astype(str)
    data_df.study_id = data_df.study_id.astype(str)
    data_df = data_df.fillna(0)
    data_df.insert(2, "path", "")
    data_df.path = data_df.subject_id.str[0:2]
    data_df.path = "p" + data_df.path
    data_df.path = data_df.path + "/p" + data_df.subject_id + "/s" + data_df.study_id + "/" + data_df.dicom_id + ".jpg"

    # process train val test df
    df = data_df
    mask1 = df[df[disease] == 1]
    mask2 = df[df['No Finding'] == 1]
    mask3 = mask1[['subject_id', 'path', disease]]
    mask4 = mask2[['subject_id', 'path', 'No Finding']]
    mask4.loc[mask4['No Finding'] == 1, 'No Finding'] = str(0)
    mask3.loc[mask3[disease] == 1, disease] = str(1)
    # balance samples
    num3 = mask3.shape[0]
    num4 = mask4.shape[0]
    if num3 < num4:
        mask4 = mask4.sample(n=num3, replace=False)

    mask3 = mask3.rename(columns={disease: 'label'})
    mask4 = mask4.rename(columns={'No Finding': 'label'})

    all_df = pd.concat([mask3, mask4], axis=0, ignore_index=True)

    train1 = mask3.sample(frac=0.8)
    train2 = mask4.sample(frac=0.8)
    mask3 = mask3[~mask3.index.isin(train1.index)]
    mask4 = mask4[~mask4.index.isin(train2.index)]
    # val1 = mask3.sample(frac=0.5)  # 0.15
    # val2 = mask4.sample(frac=0.5)
    # mask3 = mask3[~mask3.index.isin(val1.index)]
    # mask4 = mask4[~mask4.index.isin(val2.index)]

    train_data = pd.concat([train1, train2], axis=0, ignore_index=True)
    # val_data = pd.concat([val1, val2], axis=0, ignore_index=True)
    test_data = pd.concat([mask3, mask4], axis=0, ignore_index=True)

    # print(len(train_data),len(val_data),len(test_data))

    # 处理种族数据
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
    merge_df1 = merge_df1[merge_df1.race.isin(['ASIAN', 'BLACK/AFRICAN AMERICAN', 'WHITE'])]
    merge_df1 = merge_df1[merge_df1.ViewPosition.isin(['AP', 'PA'])]

    race_df = merge_df1[['subject_id', 'race']]
    race_df = race_df.drop_duplicates()

    test_data['subject_id'] = test_data['subject_id'].astype(int)
    test_data = pd.merge(test_data, race_df, how='inner', on='subject_id')
    white_test_data = test_data[test_data['race'] == 'WHITE']
    black_test_data = test_data[test_data['race'] == 'BLACK/AFRICAN AMERICAN']
    asian_test_data = test_data[test_data['race'] == 'ASIAN']

    train_data['subject_id'] = train_data['subject_id'].astype(int)
    train_data = pd.merge(train_data, race_df, how='inner', on='subject_id')
    white_train_data = train_data[train_data['race'] == 'WHITE']
    black_train_data = train_data[train_data['race'] == 'BLACK/AFRICAN AMERICAN']
    asian_train_data = train_data[train_data['race'] == 'ASIAN']

    if mode != 'all':
        if mode == 'white':
            train_data = white_train_data
        elif mode == 'black':
            train_data = black_train_data
        elif mode == 'asian':
            train_data = asian_train_data
        else:
            print('train_mode error.')
            sys.exit(0)

    print("[-------processing dataframe finish.----------]")

    return train_data, test_data, white_test_data, black_test_data, asian_test_data


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss

class EMA:
    def __init__(self, label, num_classes=None, alpha=0.9):
        self.label = label.cuda()
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))
        self.num_classes = num_classes
        self.max = torch.zeros(self.num_classes).cuda()

    def update(self, data, index, curve=None, iter_range=None, step=None):
        self.parameter = self.parameter.to(data.device)
        self.updated = self.updated.to(data.device)
        index = index.to(data.device)

        if curve is None:
            self.parameter[index] = self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        else:
            alpha = curve ** -(step / iter_range)
            self.parameter[index] = alpha * self.parameter[index] + (1 - alpha * self.updated[index]) * data
        self.updated[index] = 1

    def max_loss(self, label):
        label_index = torch.where(self.label == label)[0]
        return self.parameter[label_index].max()