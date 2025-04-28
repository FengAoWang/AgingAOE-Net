import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

PanCancer = {
    'ACC': 0, 'BLCA': 1, 'BRCA': 2, 'CESC': 3,
    'CHOL': 4, 'COADREAD': 5, 'DLBC': 6, 'ESCA': 7,
    'GBM': 8, 'HNSC': 9, 'KICH': 10, 'KIRC': 11,
    'KIRP': 12, 'LGG': 13, 'LIHC': 14, 'LUAD': 15,
    'LUSC': 16, 'MESO': 17, 'OV': 18, 'PAAD': 19,
    'PCPG': 20, 'PRAD': 21, 'SARC': 22, 'SKCM': 23,
    'STAD': 24, 'TGCT': 25, 'THCA': 26, 'THYM': 27,
    'UCEC': 28, 'UCS': 29, 'UVM': 30, '<not provided>': 31,
    'Adipose Tissue': 32, 'Adrenal Gland': 33, 'Bladder': 34, 'Blood': 35,
    'Blood Vessel': 36, 'Bone Marrow': 37, 'Brain': 38, 'Breast': 39,
    'Cervix Uteri': 40, 'Colon': 41, 'Esophagus': 42, 'Fallopian Tube': 43,
    'Heart': 44, 'Kidney': 45, 'Liver': 46, 'Lung': 47,
    'Muscle': 48, 'Nerve': 49, 'Ovary': 50, 'Pancreas': 51,
    'Pituitary': 52, 'Prostate': 53, 'Salivary Gland': 54, 'Skin': 55,
    'Small Intestine': 56, 'Spleen': 57, 'Stomach': 58, 'Testis': 59,
    'Thyroid': 60, 'Uterus': 61, 'Vagina': 62
    }

tissues = {'Adipose Tissue': 0, 'Muscle': 1, 'Blood Vessel': 2, 'Heart': 3, 'Ovary': 4, 'Uterus': 5, 'Breast': 6, 'Salivary Gland': 7, 'Brain': 8, 'Adrenal Gland': 9, 'Thyroid': 10, 'Lung': 11, 'Pancreas': 12, 'Esophagus': 13, 'Stomach': 14, 'Skin': 15, 'Colon': 16, 'Small Intestine': 17, 'Prostate': 18, 'Testis': 19, 'Nerve': 20, 'Spleen': 21, 'Pituitary': 22, 'Blood': 23, 'Vagina': 24, 'Liver': 25, 'Kidney': 26, 'Bladder': 27, 'Fallopian Tube': 28, 'Cervix Uteri': 29, '<not provided>': 30, 'Bone Marrow': 31}


def to_one_hot(num, n):
    # 创建一个n*n的单位矩阵
    eye = torch.eye(n)
    # 选择对应的行作为one-hot编码
    one_hot = eye[num]
    return one_hot


class Aging_MultiOmics_Dataset(Dataset):
    def __init__(self, clinical_info_path, multiomics_path, optional_phenotype=None, dataset=None, dataset_label=None, tissue=None):
        super(Aging_MultiOmics_Dataset, self).__init__()
        self.clinical_info = pd.read_csv(clinical_info_path)
        self.clinical_info.dropna(inplace=True, subset=['age'])

        self.clinical_info['age'].fillna(0, inplace=True)
        # 修改列名
        self.clinical_info.rename(columns={'_primary_site': 'cancer_type'}, inplace=True)

        # self.clinical_info.fillna(value=0, inplace=True)
        if optional_phenotype is not None:
            if 'pathologic_stage' in optional_phenotype:
                self.clinical_info['pathologic_stage'].fillna('none', inplace=True)
        self.multiomics_data = []
        self.multiomics_data.append(pd.read_csv(multiomics_path[0]))
        self.multiomics_data.append(pd.read_csv(multiomics_path[1], usecols=range(6619)))

        self.optional_phenotype = optional_phenotype
        self.dataset = dataset
        self.dataset_label = dataset_label
        # self.clinical_info = self.clinical_info.dropna(subset=['overall_survival'])
        # self.clinical_info.reset_index(drop=True, inplace=True)
        if tissue is not None:
            self.clinical_info = self.clinical_info[(self.clinical_info[optional_phenotype[0]].isin(tissue))]
            if 'status' in optional_phenotype:
                self.clinical_info.dropna(subset=['status'], inplace=True)
            self.clinical_info.reset_index(drop=True, inplace=True)

    def __len__(self):
        return self.clinical_info.shape[0]

    def __getitem__(self, item):
        # Get the index of the sample from the training data
        sample_id = self.clinical_info.iloc[item]['SampleID']
        if self.clinical_info.iloc[item]['age']:
            age = int(self.clinical_info.iloc[item]['age'] / 10)
        else:
            age = 0

        real_age = torch.Tensor([int(self.clinical_info.iloc[item]['age'])])
        gender = self.clinical_info.iloc[item]['gender']
        age_label = torch.LongTensor([age])
        # Get omics data for this sample
        rna_omics = self.multiomics_data[0]
        methylation_omics = self.multiomics_data[1]
        rna_item_omics = rna_omics[rna_omics['SampleID'] == sample_id].values
        methylation_item_omics = methylation_omics[methylation_omics['SampleID'] == sample_id].values
        omics_label = [True, True]
        if len(rna_item_omics) == 0:
            rna_item_omics = torch.zeros(size=[11816])
            omics_label[0] = False
        else:
            rna_item_omics = torch.Tensor(rna_item_omics.tolist()[0][2:])
            # rna_item_omics = torch.log2(rna_item_omics + 1)

        if len(methylation_item_omics) == 0:
            methylation_item_omics = torch.zeros(size=[6617])
            omics_label[1] = False
        else:
            methylation_item_omics = torch.Tensor(methylation_item_omics.tolist()[0][2:])
            # omics_label.append(1)

        omics_data = {'rna': rna_item_omics, 'methylation': methylation_item_omics}
        if self.optional_phenotype is not None:
            optional_phenotype = self.clinical_info.loc[item, self.optional_phenotype].tolist()
        else:
            optional_phenotype = ['Blood']
        dataset_label = to_one_hot(self.dataset_label, 3)
        phenotype_list = {}
        if self.optional_phenotype is not None:
            for pheno, value in zip(self.optional_phenotype, optional_phenotype):
                phenotype_list[pheno] = value
        else:
            phenotype_list['_primary_site'] = optional_phenotype[0]
        if optional_phenotype[0] in PanCancer.keys():
            tissue_label = to_one_hot(PanCancer[optional_phenotype[0]], 64)
        else:
            tissue_label = to_one_hot(PanCancer['Blood'], 64)

        # dataset_label = torch.LongTensor([self.dataset_label])
        return age_label, omics_data, phenotype_list, self.dataset, dataset_label, tissue_label, real_age, gender, omics_label, sample_id
