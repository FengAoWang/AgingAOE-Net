import os
import sys
import time
sys.path.append('/home/wfa/project/multi-omics aging')
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from dataset.dataset import Aging_MultiOmics_Dataset
from model.agingfound_model import MultiAge, ExplainMultiAge, BiTagePredictor, DeepMAgePredictor
from torch.optim import Adam
from tqdm import tqdm
import random
import numpy as np
from utils.loss_function import label_distribution_loss, improved_label_distribution_loss
import torch.nn as nn
import torch.nn.functional as F
import shap
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.model_selection import KFold


# 修改全局样式
plt.rcParams.update({
    # 'figure.figsize': (10, 6),  # 默认图表大小
    'axes.titlesize': 7,      # 坐标轴标题字体大小
    'axes.labelsize': 7,      # 坐标轴标签字体大小
    'xtick.labelsize': 6,     # x轴刻度字体大小
    'ytick.labelsize': 6,     # y轴刻度字体大小
    'legend.fontsize': 6,     # 图例字体大小
    'lines.linewidth': 0.8,      # 线条宽度
    'lines.markersize': 6,     # 标记点大小
    'axes.grid': False,         # 默认显示网格
    # 'axes.linewidth': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.size': 2,
    'xtick.major.size': 2

})

matplotlib.rcParams['pdf.fonttype'] = 42


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


set_seed(666)


tissue = 'Pancancer'



TCGA_clinical_info_path = '../data/TCGA_tumor_data/clinical_info/PanCancer_clinical_info_rna_methylation_matched.csv'
TCGA_omics_data_path = ['../data/TCGA_tumor_data/filter_data/tcga_rna_gene_tpm_filter_2.csv',
                        '../data/TCGA_tumor_data/filter_data/TCGA_DNA_methylation_matched_filter.csv']
TCGA_PanCancer_dataset = Aging_MultiOmics_Dataset(TCGA_clinical_info_path, TCGA_omics_data_path, ['cancer_type', 'overall_survival', 'status'],
                                                  dataset='TCGA PanCancer', dataset_label=0)

merge_dataset = ConcatDataset([TCGA_PanCancer_dataset])


def agingFound_predict(train_loader, test_loader, fold_id):
    torch.cuda.set_device(6)
    epochs = 10
    multiAge = MultiAge([11816, 6617],  128,
                        'model_dict/pretrained_agingFound_epoch50_randomSeed_3407_latentDim128.pt')
    multiAge = multiAge.cuda()
    optimizer = Adam(multiAge.parameters(), lr=0.001, weight_decay=1e-4)
    loss_function = nn.MSELoss()
    multiAge.train()
    start_time = time.time()
    for epoch in range(epochs):
        print(f'-------epoch{epoch}--------')
        train_total_loss = 0

        with tqdm(train_loader, unit='batch') as tepoch:
            for batch, data in enumerate(tepoch):
                age_label, omics_data, _, _, dataset_label, tissue_label, real_age, _, _, _ = data
                omics_input = [omics_data['rna'].cuda(), omics_data['methylation'].cuda()]
                dataset_label = dataset_label.cuda()
                tissue_label = tissue_label.cuda()
                real_age = real_age.cuda()
                real_age = real_age.squeeze()
                age_label = age_label.cuda()
                multi_predict_logit = multiAge(omics_input, dataset_label, tissue_label, [True, True])

                loss = improved_label_distribution_loss(multi_predict_logit, real_age)
                train_total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f'TrainTotal loss: {train_total_loss / len(train_loader)}')

    # 测试阶段
    multiAge.eval()
    fold_results = {
        'real_age': [],
        'predict_age': [],
        'age_gap': [],
        'os': [],
        'status': [],
        'cancer_type': [],
        'gender': [],
        'SampleID': []
    }

    test_total_rna_loss = 0

    with torch.no_grad():
        with tqdm(test_loader, unit='batch') as tepoch:
            for batch, data in enumerate(tepoch):
                age_label, omics_data, phenotype_list, _, dataset_label, tissue_label, real_age, gender, _, SampleID = data
                omics_input = [omics_data['rna'].cuda(), omics_data['methylation'].cuda()]
                dataset_label = dataset_label.cuda()
                tissue_label = tissue_label.cuda()
                real_age = real_age.cuda()
                multi_predict_age = multiAge.age_predict(omics_input, dataset_label, tissue_label, [True, True])

                fold_results['real_age'].extend(torch.squeeze(real_age, dim=1).cpu().numpy())
                fold_results['predict_age'].extend(multi_predict_age.squeeze(dim=1).detach().cpu().numpy())
                fold_results['age_gap'].extend((multi_predict_age - real_age).squeeze(dim=1).detach().cpu().numpy())
                fold_results['os'].extend(phenotype_list['overall_survival'].numpy())
                fold_results['status'].extend(phenotype_list['status'].numpy())
                fold_results['cancer_type'].extend(phenotype_list['cancer_type'])
                fold_results['gender'].extend(gender)
                fold_results['SampleID'].extend(SampleID)

                rna_loss = loss_function(real_age, multi_predict_age)
                test_total_rna_loss += rna_loss.item()

    print(f'TestTotal loss: {test_total_rna_loss / len(test_loader)}')
    fold_df = pd.DataFrame(fold_results)
    print(fold_df)

    print(f'time used: {time.time() - start_time}')
    torch.save(multiAge.state_dict(), f'model_dict/multiAge_{tissue}_fold{fold_id}.pt')
    return fold_df


def train_DeepMAge(train_loader, test_loader, fold_id):
    torch.cuda.set_device(1)
    epochs = 20
    multiAge = DeepMAgePredictor([11816, 6617])
    multiAge = multiAge.cuda()
    optimizer = Adam(multiAge.parameters(), lr=0.0001, weight_decay=1e-3)
    loss_function = nn.L1Loss()

    multiAge.train()
    for epoch in range(epochs):
        print(f'-------epoch{epoch}--------')
        train_total_loss = 0
        start_time = time.time()
        with tqdm(train_loader, unit='batch') as tepoch:
            for batch, data in enumerate(tepoch):
                age_label, omics_data, _, _, dataset_label, tissue_label, real_age, _, _, _ = data
                omics_input = [omics_data['rna'].cuda(), omics_data['methylation'].cuda()]
                real_age = real_age.cuda()
                predict_age = multiAge(omics_input)
                mae_loss = loss_function(predict_age, real_age)
                loss = mae_loss

                train_total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f'TrainTotal loss: {train_total_loss / len(train_dataloader)}')

        # 测试阶段
        multiAge.eval()
        fold_results = {
            'real_age': [],
            'predict_age': [],
            'age_gap': [],
            'os': [],
            'status': [],
            'cancer_type': [],
            'gender': [],
            'SampleID': []
        }

        test_total_rna_loss = 0

        with torch.no_grad():
            with tqdm(test_loader, unit='batch') as tepoch:
                for batch, data in enumerate(tepoch):
                    age_label, omics_data, phenotype_list, _, dataset_label, tissue_label, real_age, gender, _, SampleID = data
                    omics_input = [omics_data['rna'].cuda(), omics_data['methylation'].cuda()]
                    real_age = real_age.cuda()
                    multi_predict_age = multiAge(omics_input)

                    fold_results['real_age'].extend(torch.squeeze(real_age, dim=1).cpu().numpy())
                    fold_results['predict_age'].extend(multi_predict_age.squeeze(dim=1).detach().cpu().numpy())
                    fold_results['age_gap'].extend((multi_predict_age - real_age).squeeze(dim=1).detach().cpu().numpy())
                    fold_results['os'].extend(phenotype_list['overall_survival'].numpy())
                    fold_results['status'].extend(phenotype_list['status'].numpy())
                    fold_results['cancer_type'].extend(phenotype_list['cancer_type'])
                    fold_results['gender'].extend(gender)
                    fold_results['SampleID'].extend(SampleID)

                    rna_loss = loss_function(real_age, multi_predict_age)
                    test_total_rna_loss += rna_loss.item()

        print(f'TestTotal loss: {test_total_rna_loss / len(test_loader)}')
        fold_df = pd.DataFrame(fold_results)
        print(fold_df)

        print(f'time used: {time.time() - start_time}')
        torch.save(multiAge.state_dict(), f'model_dict/BiTage_{tissue}_fold{fold_id}.pt')
        return fold_df

# agingFound_training()

def train_BiTMAge(train_loader, test_loader, fold_id):
    torch.cuda.set_device(1)
    epochs = 20
    multiAge = BiTagePredictor([11816, 6617])
    multiAge = multiAge.cuda()
    optimizer = Adam(multiAge.parameters(), lr=0.0001)
    loss_function = nn.L1Loss()
    alpha = 1.0
    l1_ratio = 0.5

    multiAge.train()
    for epoch in range(epochs):
        print(f'-------epoch{epoch}--------')
        train_total_loss = 0
        start_time = time.time()
        with tqdm(train_loader, unit='batch') as tepoch:
            for batch, data in enumerate(tepoch):
                age_label, omics_data, _, _, dataset_label, tissue_label, real_age, _, _, _ = data
                omics_input = [omics_data['rna'].cuda(), omics_data['methylation'].cuda()]
                real_age = real_age.cuda()
                predict_age = multiAge(omics_input)

                mae_loss = loss_function(predict_age, real_age)
                l1_norm = torch.norm(multiAge.RNAEncoder.weight, p=1)
                l2_norm = torch.norm(multiAge.RNAEncoder.weight, p=2)
                loss = mae_loss + alpha * l1_ratio * l1_norm + 0.5 * alpha * (1.0 - l1_ratio) * l2_norm

                train_total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f'TrainTotal loss: {train_total_loss / len(train_dataloader)}')

        # 测试阶段
        multiAge.eval()
        fold_results = {
            'real_age': [],
            'predict_age': [],
            'age_gap': [],
            'os': [],
            'status': [],
            'cancer_type': [],
            'gender': [],
            'SampleID': []
        }

        test_total_rna_loss = 0

        with torch.no_grad():
            with tqdm(test_loader, unit='batch') as tepoch:
                for batch, data in enumerate(tepoch):
                    age_label, omics_data, phenotype_list, _, dataset_label, tissue_label, real_age, gender, _, SampleID = data
                    omics_input = [omics_data['rna'].cuda(), omics_data['methylation'].cuda()]
                    real_age = real_age.cuda()
                    multi_predict_age = multiAge(omics_input)

                    fold_results['real_age'].extend(torch.squeeze(real_age, dim=1).cpu().numpy())
                    fold_results['predict_age'].extend(multi_predict_age.squeeze(dim=1).detach().cpu().numpy())
                    fold_results['age_gap'].extend((multi_predict_age - real_age).squeeze(dim=1).detach().cpu().numpy())
                    fold_results['os'].extend(phenotype_list['overall_survival'].numpy())
                    fold_results['status'].extend(phenotype_list['status'].numpy())
                    fold_results['cancer_type'].extend(phenotype_list['cancer_type'])
                    fold_results['gender'].extend(gender)
                    fold_results['SampleID'].extend(SampleID)

                    rna_loss = loss_function(real_age, multi_predict_age)
                    test_total_rna_loss += rna_loss.item()

        print(f'TestTotal loss: {test_total_rna_loss / len(test_loader)}')
        fold_df = pd.DataFrame(fold_results)
        print(fold_df)

        print(f'time used: {time.time() - start_time}')
        torch.save(multiAge.state_dict(), f'model_dict/BiTage_{tissue}_fold{fold_id}.pt')
        return fold_df


def get_sample_data(dataloader, num_samples=4):
    all_data = torch.Tensor([]).cuda()
    with tqdm(dataloader, unit='batch') as tepoch:
        for batch, data in enumerate(tepoch):
            age_label, omics_data, phenotype_list, _, dataset_label, tissue_label, real_age, _, _, _ = data
            omics_input = [omics_data['rna'].cuda(), omics_data['methylation'].cuda()]
            dataset_label = dataset_label.cuda()
            tissue_label = tissue_label.cuda()
            real_age = real_age.cuda()
            explain_input_x = [omics_input[0], omics_input[1], dataset_label, tissue_label]
            explain_input_x = torch.concat(explain_input_x, dim=1)
            # explain_input_x = (*explain_input_x,)
            all_data = torch.cat((all_data, explain_input_x), dim=0)

            if all_data.shape[0] >= num_samples:
                break

        return all_data


def model_explain(train_loader, test_loader, fold_id):
    torch.cuda.set_device(6)
    explainMultiAge_model = ExplainMultiAge([11816, 6617],128,
                            f'model_dict/multiAge_{tissue}_fold{fold_id}.pt',
                            [True, True])
    explainMultiAge_model.eval()
    explainMultiAge_model.cuda()
    TCGA_omics_data_path = ['../data/TCGA_tumor_data/filter_data/tcga_rna_gene_tpm_filter_2.csv',
                            '../data/TCGA_tumor_data/filter_data/TCGA_DNA_methylation_matched_filter.csv']
    gene_ids = pd.read_csv(TCGA_omics_data_path[0]).columns.tolist()[2:]

    methylation_ids = pd.read_csv(TCGA_omics_data_path[1]).columns.tolist()[2:]
    all_data = get_sample_data(train_loader, num_samples=100)
    # 创建SHAP解释器
    new_data = get_sample_data(test_loader, num_samples=100)

    start_time = time.time()
    explainer = shap.DeepExplainer(explainMultiAge_model, all_data)

    # with torch.no_grad():
    shap_values = explainer.shap_values(new_data, check_additivity=False)
    print(shap_values.shape)
    all_shap_values = shap.Explanation(shap_values[:, :11816+6617], feature_names=gene_ids + methylation_ids)

    rna_shap_values = shap.Explanation(shap_values[:, :11816], feature_names=gene_ids)

    methylation_shap_values = shap.Explanation(shap_values[:, 11816:11816+6617], feature_names=methylation_ids)
    print('time used: ', time.time() - start_time)
    # summarize the effects of all the features

    plt.figure(figsize=(2.5, 4))
    shap.summary_plot(rna_shap_values, new_data[:, :11816].detach().cpu().numpy(), show=False, max_display=10)
    # 获取当前图像并保存
    plt.savefig('rna_shap_summary_plot.pdf', dpi=1000)  # 你可以更改文件名和 DPI（分辨率）
    plt.close()  # 关闭图像，避免后续绘图出现重叠

    gene_shap_df = pd.DataFrame({
        'shap_value': np.mean(np.abs(rna_shap_values.values), axis=0),
        'gene_name': gene_ids},
    )
    gene_shap_df.to_csv(f'gene_shap_summary_fold{fold_id}.csv')

    plt.figure(figsize=(2.5, 4))
    shap.summary_plot(methylation_shap_values, new_data[:, 11816:11816+6617].detach().cpu().numpy(), show=False, max_display=10)
    # 获取当前图像并保存
    plt.savefig('methylation_shap_summary_plot.pdf', dpi=1000)  # 你可以更改文件名和 DPI（分辨率）
    plt.close()  # 关闭图像，避免后续绘图出现重叠

    methylation_shap_df = pd.DataFrame({'shap_value': np.mean(np.abs(methylation_shap_values.values), axis=0),
                                        'cpg_sits': methylation_ids})
    methylation_shap_df.to_csv(f'methylation_shap_summary_fold{fold_id}.csv')

    # ========== 选取 RNA SHAP 值最高的 10 个基因特征 ==========
    # rna_shap_values.values 的形状一般为 (样本数, RNA 特征数)
    # 计算每个基因的平均绝对 SHAP 值
    rna_mean_abs_shap = np.mean(np.abs(rna_shap_values.values), axis=0)  # (RNA 特征数,)

    # 选出最重要（平均绝对值最高）的前 10 个特征下标
    top10_indices = np.argsort(rna_mean_abs_shap)[-10:]  # 从小到大排序，取后 10 个


    methylation_mean_abs_shap = np.mean(np.abs(methylation_shap_values.values), axis=0)
    methy_top10_indices = np.argsort(methylation_mean_abs_shap)[-10:]

    # 打印这 10 个基因的名称
    print("Top 10 RNA features by absolute SHAP value:")
    for idx in reversed(top10_indices):  # 如果想从大到小打印，可以再反转一次
        print(f"  Feature index: {idx}, Gene: {gene_ids[idx]}, Mean|SHAP|: {rna_mean_abs_shap[idx]:.4f}")

    # 利用 shap_visualization() 函数对这 10 个基因可视化
    # 假设 shap_visualization() 的第一个参数是特征值，第二个是 shap 值
    for idx in reversed(top10_indices):
        gene_name = gene_ids[idx]
        print(f"Visualizing SHAP for gene {gene_name} (index={idx})...")
        shap_visualization(
            new_data[:, idx].detach().cpu().numpy(),  # 指定该基因在所有样本上的输入值
            shap_values[:, idx],
            gene_ids[idx]
        )

    for idx in reversed(methy_top10_indices):
        cpg_name = methylation_ids[idx]
        print(f"Visualizing SHAP for gene {cpg_name} (index={idx})...")
        shap_visualization(
            new_data[:, 11816+idx].detach().cpu().numpy(),  # 指定该基因在所有样本上的输入值
            shap_values[:, 11816+idx],
            methylation_ids[idx]
        )

def shap_visualization(omics_data, omics_shap, gene_name):
    fig = plt.figure(figsize=(3, 2), dpi=1200)
    grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)
    # 主散点图
    main_ax = fig.add_subplot(grid[1:, :-1])
    # 绘制散点图
    main_ax.scatter(omics_data, omics_shap, s=10, alpha=0.8, color="#6A9ACE")
    # 添加拟合线
    sns.regplot(
        x=omics_data,
        y=omics_shap,
        scatter=False, # 不绘制散点，仅绘制拟合线
        lowess=True,    # 使用 LOWESS 曲线进行拟合
        color="#6A9ACE",
        ax=main_ax)
    main_ax.axhline(y=0, color='black', linestyle='-.', linewidth=1)
    main_ax.set_xlabel(f'{gene_name}', fontsize=6)
    main_ax.set_ylabel(f'SHAP value for {gene_name}', fontsize=6)
    main_ax.spines['top'].set_visible(False)
    main_ax.spines['right'].set_visible(False)
    # 顶部X轴边缘分布
    top_ax = fig.add_subplot(grid[0, :-1], sharex=main_ax)
    sns.kdeplot(omics_data, ax=top_ax, fill=True, color="#6A9ACE")
    top_ax.axis('off')
    # 右侧Y轴边缘分布
    right_ax = fig.add_subplot(grid[1:, -1], sharey=main_ax)
    sns.kdeplot(y=omics_shap, ax=right_ax, fill=True, color="#6A9ACE")
    right_ax.axis('off')
    # 保存图表
    plt.savefig(f"{gene_name}_shap_fold{fold_idx}.pdf", format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    # gene_ids = [8519, 7263, 5062, 3256, 5269, 9204, 2859, 11355, 2743]
    #
    # for gene_id in gene_ids:
    #     target_gene_visulaize(gene_id)
    # 设置五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=666)
    all_test_dfs = []
    all_DeepMage_df = []
    all_BiTage_df = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(merge_dataset)):
        print(f"\n=========== Processing Fold {fold_idx + 1}/5 ===========")

        # 创建当前fold的数据集
        train_dataset = Subset(merge_dataset, train_idx)
        test_dataset = Subset(merge_dataset, test_idx)

        # 创建数据加载器
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        all_DeepMage_df.append(train_DeepMAge(train_dataloader, test_dataloader, fold_idx))
        all_BiTage_df.append(train_BiTMAge(train_dataloader, test_dataloader, fold_idx))
        result_fold = agingFound_predict(train_dataloader, test_dataloader, fold_idx)
        model_explain(train_dataloader, test_dataloader, fold_idx)
        all_test_dfs.append(result_fold)


    # 合并所有fold结果
    final_df = pd.concat(all_test_dfs, ignore_index=True)
    final_df.to_csv(f'agingFound_Age/test_age_multiomics_{tissue}_phenotype.csv', index=False)

    DeepMage_final_df = pd.concat(all_DeepMage_df, ignore_index=True)
    DeepMage_final_df.to_csv(f'agingFound_Age/test_DeepMage_multiomics_{tissue}_phenotype.csv', index=False)

    BiTage_final_df = pd.concat(all_BiTage_df, ignore_index=True)
    BiTage_final_df.to_csv(f'agingFound_Age/test_BiTage_multiomics_{tissue}_phenotype.csv', index=False)


    print("All folds results saved!")

