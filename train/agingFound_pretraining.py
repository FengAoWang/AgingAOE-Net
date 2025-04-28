import os
import sys
import time
sys.path.append('/home/wfa/project/multiomics_aging')
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
from dataset.dataset import Aging_MultiOmics_Dataset
from model.agingfound_model import AgingFound, ModalVAE
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import random
import numpy as np
import anndata
import scanpy as sc


def set_seed(seed):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_seed = 3407
set_seed(random_seed)

heart_disease_clinical_info_path = '../data/Heart_disease_data/clinical_info/GSE221615_clinical_infos.csv'
heart_disease_omics_path = ['../data/Heart_disease_data/filter_data/GSE221615_rna_normalizedCounts_filter_2.csv',
                            '../data/TCGA_tumor_data/filter_data/TCGA_DNA_methylation_matched_filter.csv']

heart_disease_dataset = Aging_MultiOmics_Dataset(heart_disease_clinical_info_path, heart_disease_omics_path, dataset='heart disease', dataset_label=2)


GTEx_clinical_info_path = '../data/GTEX_normal_data/clinical_info/GTEx_sample_clinical_info_matched_gender.csv'
GTEx_omics_data_path = ['../data/GTEX_normal_data/filter_data/gtex_rna_gene_tpm_filter_2.csv',
                        '../data/TCGA_tumor_data/filter_data/TCGA_DNA_methylation_matched_filter.csv']
GTEx_normal_dataset = Aging_MultiOmics_Dataset(GTEx_clinical_info_path, GTEx_omics_data_path, dataset='GTEx normal', dataset_label=1)

TCGA_clinical_info_path = '../data/TCGA_tumor_data/clinical_info/PanCancer_clinical_info_rna_methylation_matched.csv'
TCGA_omics_data_path = ['../data/TCGA_tumor_data/filter_data/tcga_rna_gene_tpm_filter_2.csv',
                        '../data/TCGA_tumor_data/filter_data/TCGA_DNA_methylation_matched_filter.csv']
TCGA_PanCancer_dataset = Aging_MultiOmics_Dataset(TCGA_clinical_info_path, TCGA_omics_data_path, dataset='TCGA PanCancer', dataset_label=0)
# optional_phenotype=['cancer_type']
merge_dataset = ConcatDataset([GTEx_normal_dataset, TCGA_PanCancer_dataset, heart_disease_dataset])
merge_dataloader = DataLoader(merge_dataset, batch_size=256, shuffle=True, num_workers=8)

TCGA_dataloader = DataLoader(TCGA_PanCancer_dataset, batch_size=256, shuffle=True, num_workers=8)

epochs = 50
latent_dim = 128


def agingFound_pretraining():
    torch.cuda.set_device(6)

    agingFound = AgingFound([11816, 6617], latent_dim=latent_dim)
    # agingFound_dict = torch.load(f'model_dict/pretrained_agingFound_epoch{epochs}_randomSeed_{random_seed}_latentDim128.pt', map_location='cpu')
    # agingFound.load_state_dict(agingFound_dict)
    agingFound = agingFound.cuda()

    agingFound_optimizer = Adam(agingFound.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = StepLR(agingFound_optimizer, step_size=20, gamma=0.1)

    # lr 0.00001

    agingFound.train()
    for epoch in range(epochs):
        print(f'-------epoch{epoch}--------')
        total_rna_pretrain_loss = 0
        start_time = time.time()
        with tqdm(merge_dataloader, unit='batch') as tepoch:
            for batch, data in enumerate(tepoch):
                age_label, omics_data, _, _, dataset_label, tissue_label, _, _, omics_label, _ = data
                age_label = age_label.cuda()
                omics_input = [omics_data['rna'].cuda(), omics_data['methylation'].cuda()]
                dataset_label = dataset_label.cuda()
                tissue_label = tissue_label.cuda()
                loss, _ = agingFound.compute_loss(omics_input, dataset_label, tissue_label, [True, False], age_label)
                # output = agingFound(omics_input, dataset_label, tissue_label)
                agingFound_optimizer.zero_grad()
                loss.backward()
                agingFound_optimizer.step()
                total_rna_pretrain_loss += loss.item()
        print(f'rna pretrain loss: {total_rna_pretrain_loss / len(merge_dataloader)}')
        print(f'time used: {time.time() - start_time}')

    agingFound.train()
    # agingFound.con_weight = 10
    for epoch in range(epochs):
        print(f'-------epoch{epoch}--------')
        total_rna_pretrain_loss = 0

        start_time = time.time()

        total_multi_pretrain_loss = 0
        total_age_loss = 0
        with tqdm(TCGA_dataloader, unit='batch') as tepoch:
            for batch, data in enumerate(tepoch):
                age_label, omics_data, _, _, dataset_label, tissue_label, _, _, omics_label, _ = data
                age_label = age_label.cuda()
                omics_input = [omics_data['rna'].cuda(), omics_data['methylation'].cuda()]
                dataset_label = dataset_label.cuda()
                tissue_label = tissue_label.cuda()
                loss, age_loss = agingFound.compute_loss(omics_input, dataset_label, tissue_label, [True, True], age_label)
                # output = agingFound(omics_input, dataset_label, tissue_label)
                agingFound_optimizer.zero_grad()
                loss.backward()
                agingFound_optimizer.step()
                total_multi_pretrain_loss += loss.item()
                total_age_loss += age_loss.item()
                # tepoch.set_postfix(loss=loss.item(), recon_loss=recon_loss.item(), kl_loss=kl_loss, ageContrastive_loss = ageContrastive_loss)

        print(f'total pretrain loss: {total_multi_pretrain_loss / len(TCGA_dataloader)}')
        print(f'total age loss: {total_age_loss / len(TCGA_dataloader)}')
        print(f'time used: {time.time() - start_time}')

    model_dict = agingFound.state_dict()
    torch.save(model_dict, f'model_dict/pretrained_agingFound_epoch{epochs}_randomSeed_{random_seed}_latentDim{latent_dim}.pt')


agingFound_pretraining()
