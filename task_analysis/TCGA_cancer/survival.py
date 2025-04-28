import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import pandas as pd
from lifelines.statistics import logrank_test, multivariate_logrank_test
import matplotlib
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.linalg import lstsq
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu

# 修改全局样式
plt.rcParams.update({
    # 'figure.figsize': (10, 6),  # 默认图表大小
    'axes.titlesize': 7,  # 坐标轴标题字体大小
    'axes.labelsize': 7,  # 坐标轴标签字体大小
    'xtick.labelsize': 6,  # x轴刻度字体大小
    'ytick.labelsize': 6,  # y轴刻度字体大小
    'legend.fontsize': 6,  # 图例字体大小
    'lines.linewidth': 0.5,  # 线条宽度
    'lines.markersize': 6,  # 标记点大小
    'axes.grid': False,  # 默认显示网格
    # 'axes.linewidth': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.size': 2,
    'xtick.major.size': 2,
    'pdf.fonttype': 42,
    'ps.fonttype': 42

})

tissue = 'Pancancer'


# test_age_Lasso_phenotype.csv
# test_age_multiomics_{cancer}_phenotype.csv


def survival_analysis(survival_df, path):
    # 按真实年龄排序
    survival_df.sort_values(by=['real_age'], inplace=True)

    # 计算LOESS平滑曲线
    predict_mean = sm.nonparametric.lowess(survival_df['predict_age'], survival_df['real_age'])

    # 过滤掉预测年龄为负数的数据
    survival_df = survival_df[survival_df['predict_age'] > 0]

    # 根据 loess_age_gap 划分三组
    survival_df.loc[survival_df['loess_age_gap'] <= -5, 'group'] = 0  # Decelerated
    survival_df.loc[(survival_df['loess_age_gap'] > -5) & (survival_df['loess_age_gap'] <= 5), 'group'] = 1  # Stable
    survival_df.loc[survival_df['loess_age_gap'] > 5, 'group'] = 2 # Accelerated

    # 转换生存时间单位（年）
    survival_df['os'] = survival_df['os'] / 365

    # 将超过 5 年的生存时间设为 5
    survival_df.loc[survival_df['os'] > 5, 'status'] = 0
    survival_df.loc[survival_df['os'] > 5, 'os'] = 5

    # 按组提取数据
    group0 = survival_df[survival_df['group'] == 0]  # Decelerated
    group1 = survival_df[survival_df['group'] == 1]  # Accelerated
    group2 = survival_df[survival_df['group'] == 2]  # Stable

    # 创建 KaplanMeierFitter 对象
    kmf = KaplanMeierFitter()

    # 计算三组之间的log-rank p值
    results = multivariate_logrank_test(survival_df['os'], survival_df['group'], survival_df['status'])
    p_value = results.p_value
    print("Multivariate Log-Rank p-value:", p_value)

    # 计算 Harrell’s C-index
    c_index = concordance_index(survival_df['os'], -survival_df['loess_age_gap'], event_observed=survival_df['status'])
    print("Concordance Index:", c_index)

    # 计算五年生存 AUROC
    auroc = roc_auc_score(survival_df['status'], survival_df['loess_age_gap'])
    print("Five-Year Survival AUROC:", auroc)

    # 画生存曲线
    plt.figure(figsize=(1.6, 1.6))

    kmf.fit(group0['os'], group0['status'], label='Decelerated')
    ax = kmf.plot()

    kmf.fit(group1['os'], group1['status'], label='Normal')
    kmf.plot(ax=ax)

    kmf.fit(group2['os'], group2['status'], label='Accelerated')
    kmf.plot(ax=ax)

    # 添加图标题和坐标轴标签
    plt.title('Survival Curve by Age Gap')
    plt.xlabel('Survival years')
    plt.ylabel('Survival Probability')

    # 在图上标注 p 值
    plt.text(0.5, 0.6, f'p-value = {p_value:.2e}', fontsize=7)

    # 保存图片
    plt.savefig(path, dpi=1000, bbox_inches='tight')

    return p_value, c_index, auroc


def Pancancer_HR_analysis(survival_df, cancer_types, path):
    # 用于存储结果的列表
    results = []

    for cancer_type in cancer_types:
        print(f"正在处理癌症类型: {cancer_type}")

        # 筛选当前癌症类型的数据
        cancer_df = survival_df[survival_df['cancer_type'] == cancer_type]

        # mean_predict_age_per_real_age = cancer_df.groupby('real_age')['predict_age'].transform('median')
        # predict_age = cancer_df['predict_age']
        # cancer_df['loess_age_gap'] = -(mean_predict_age_per_real_age - predict_age)

        # 按年龄排序
        cancer_df.sort_values(by=['real_age'], inplace=True)

        # 平滑拟合
        predict_mean = sm.nonparametric.lowess(cancer_df['predict_age'], cancer_df['real_age'])
        cancer_df = cancer_df[cancer_df['predict_age'] > 0]

        # 根据 loess_age_gap 分组
        cancer_df.loc[cancer_df['loess_age_gap'] <= 0, 'group'] = 0
        cancer_df.loc[cancer_df['loess_age_gap'] > 0, 'group'] = 1

        # 处理生存时间
        cancer_df['os'] = cancer_df['os'] / 365
        cancer_df.loc[cancer_df['os'] > 5, 'status'] = 0
        cancer_df.loc[cancer_df['os'] > 5, 'os'] = 5

        # 构建 Cox 模型数据
        data = cancer_df[['os', 'status', 'loess_age_gap']]

        # CoxPH 模型拟合
        cph = CoxPHFitter()
        cph.fit(data, duration_col='os', event_col='status')

        # 提取 HR 和置信区间
        desired_factor = 'loess_age_gap'
        coef = cph.summary.loc[desired_factor, 'coef']
        se = cph.summary.loc[desired_factor, 'se(coef)']
        hazard_ratio = np.exp(coef)
        lower_ci = np.exp(coef - 1.96 * se)
        upper_ci = np.exp(coef + 1.96 * se)
        p_values = cph.summary.loc[desired_factor, 'p']

        # 计算 q 值
        _, q_values, _, _ = multipletests([p_values], method='fdr_bh')

        # 存储结果
        results.append({
            'cancer_type': cancer_type,
            'HR': hazard_ratio,
            'Lower 95% CI': lower_ci,
            'Upper 95% CI': upper_ci,
            'P-value': p_values,
            'Q-value': q_values[0]
        })

        # 打印结果
        print(f"{cancer_type} 的 HR 值: {hazard_ratio}")
        print(f"{cancer_type} 的 95% 置信区间: ({lower_ci}, {upper_ci})")
        print(f"{cancer_type} 的 P 值: {p_values}, Q 值: {q_values[0]}")

    # 将结果转换为 DataFrame
    results_df = pd.DataFrame(results)

    # 可视化 HR 值
    plt.figure(figsize=(2.5, 2))
    plt.errorbar(
        results_df['HR'], results_df['cancer_type'],
        xerr=[
            results_df['HR'] - results_df['Lower 95% CI'],
            results_df['Upper 95% CI'] - results_df['HR']
        ],
        fmt='o',  # 中间点的形状
        markersize=6,  # 中间点大小
        markerfacecolor='#6784C2',  # 中间点填充颜色
        markeredgecolor='black',  # 中间点边框颜色
        markeredgewidth=0.5,
        ecolor='black',  # 误差棒主干颜色
        elinewidth=0.5,  # 误差棒主干宽度
        capsize=0,  # 误差棒两端横线长度
        capthick=1,  # 误差棒两端横线厚度
    )
    plt.axvline(x=1, color='gray', linestyle='--', label='HR=1')

    # 在 y 轴最右侧标注 P-value
    x_max = results_df['Upper 95% CI'].max() + 0.03  # 找到最大 X 值用于定位
    for idx, row in results_df.iterrows():
        plt.text(
            x_max, row['cancer_type'],  # 调整 x 位置以对齐到右侧
            f"{row['P-value']:.2e}",
            va='center', ha='left', fontsize=6,
        )

    plt.xlabel('Hazard Ratio (HR)')
    plt.ylabel('Cancer Type')
    plt.title('Hazard Ratios by Cancer Type with P-values')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=1000, bbox_inches='tight')

    return results_df


def Method_HR_analysis(survival_df_list, method_list, item_list, path):
    # 用于存储结果的列表
    results = []

    # 不同方法对应的形状
    markers = ['o', 'p', 'h', 's', 'v', '>', '<', '*', 'D', '^']  # 可自行扩展或调整顺序
    marker_dict = {method: markers[i % len(markers)] for i, method in enumerate(method_list)}

    for i in range(len(survival_df_list)):
        print(f"正在处理方法: {method_list[i]}")

        # 筛选当前癌症类型的数据

        # 按年龄排序
        cancer_df = survival_df_list[i]

        cancer_df = cancer_df[cancer_df['predict_age'] > 0]

        # 根据 loess_age_gap 分组
        cancer_df.loc[cancer_df['loess_age_gap'] <= 0, 'group'] = 0
        cancer_df.loc[cancer_df['loess_age_gap'] > 0, 'group'] = 1

        # 处理生存时间
        cancer_df['os'] = cancer_df['os'] / 365
        cancer_df.loc[cancer_df['os'] > 5, 'status'] = 0
        cancer_df.loc[cancer_df['os'] > 5, 'os'] = 5

        # 构建 Cox 模型数据
        data = cancer_df[['os', 'status', item_list[i]]]

        # CoxPH 模型拟合
        cph = CoxPHFitter()
        cph.fit(data, duration_col='os', event_col='status')

        # 提取 HR 和置信区间
        desired_factor = item_list[i]
        coef = cph.summary.loc[desired_factor, 'coef']
        se = cph.summary.loc[desired_factor, 'se(coef)']
        hazard_ratio = np.exp(coef)
        lower_ci = np.exp(coef - 1.96 * se)
        upper_ci = np.exp(coef + 1.96 * se)
        p_values = cph.summary.loc[desired_factor, 'p']

        # 计算 q 值
        _, q_values, _, _ = multipletests([p_values], method='fdr_bh')

        # 存储结果
        results.append({
            'method': method_list[i],
            'HR': hazard_ratio,
            'Lower 95% CI': lower_ci,
            'Upper 95% CI': upper_ci,
            'P-value': p_values,
            'Q-value': q_values[0]
        })

        # 打印结果
        print(f"{method_list[i]} 的 HR 值: {hazard_ratio}")
        print(f"{method_list[i]} 的 95% 置信区间: ({lower_ci}, {upper_ci})")
        print(f"{method_list[i]} 的 P 值: {p_values}, Q 值: {q_values[0]}")

    # 将结果转换为 DataFrame
    results_df = pd.DataFrame(results)

    # 可视化 HR 值
    plt.figure(figsize=(2.5, 2))
    palette = sns.color_palette("Paired", len(method_list))  # 可以选择其他调色板，如 "viridis", "coolwarm"
    colors = palette

    for i, method in enumerate(method_list):
        # 获取当前方法的形状
        marker = marker_dict[method]
        row = results_df[results_df['method'] == method].iloc[0]

        plt.errorbar(
            row['HR'], method,  # 横坐标和纵坐标
            xerr=[[row['HR'] - row['Lower 95% CI']], [row['Upper 95% CI'] - row['HR']]],
            fmt=marker,
            markersize=6,  # 中间点大小
            markerfacecolor=colors[i],  # 中间点填充颜色
            markeredgecolor='black',  # 中间点边框颜色
            markeredgewidth=0.5,
            ecolor='black',  # 误差棒主干颜色
            elinewidth=0.5,  # 误差棒主干宽度
            capsize=0,  # 误差棒两端横线长度
            capthick=1,  # 误差棒两端横线厚度
            label=method  # 使用不同的形状
        )
    plt.axvline(x=1, color='gray', linestyle='--', label='HR=1')

    # 在 y 轴最右侧标注 P-value
    x_max = results_df['Upper 95% CI'].max()  # 找到最大 X 值用于定位
    for idx, row in results_df.iterrows():
        plt.text(
            x_max, row['method'],  # 调整 x 位置以对齐到右侧
            f"{row['P-value']:.2e}",
            va='center', ha='left', fontsize=6,
        )

    plt.xlabel('Hazard Ratio (HR)')
    plt.ylabel('Cancer Type')
    plt.title('Hazard Ratios by Cancer Type with P-values')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=1000, bbox_inches='tight')
    return results_df


def calculate_auc_at_five_years(cancer_df, item):
    """
    计算五年生存预测的 AUC
    """
    # 二分类标签：生存是否超过五年
    cancer_df['five_year_survival'] = (cancer_df['os'] > 10).astype(int)

    # 预测概率（这里假设使用 predict_age 作为风险评分）
    y_true = cancer_df['five_year_survival']
    y_pred = cancer_df[item]

    # 计算 AUC
    auc = roc_auc_score(y_true, -y_pred)
    return auc


def Method_c_index_analysis(survival_df_list, method_list, item_list, path):
    results = []
    c_index_list = []
    auc_list = []

    for i in range(len(survival_df_list)):
        print(f"正在处理方法: {method_list[i]}")
        cancer_df = survival_df_list[i]

        # 筛选数据
        cancer_df = cancer_df[cancer_df['predict_age'] > 0]

        # 转换生存时间为年
        cancer_df['os'] = cancer_df['os'] / 365

        # 计算 C-index
        c_index = concordance_index(cancer_df['os'], -cancer_df[item_list[i]], cancer_df['status'])
        c_index_list.append(c_index)

        # 计算五年生存 AUC
        auc = calculate_auc_at_five_years(cancer_df, item_list[i])
        auc_list.append(auc)

        # 保存结果
        results.append({
            "method": method_list[i],
            "c_index": c_index,
            "auc_five_year": auc
        })

    # 转换结果为 DataFrame

    # 保存结果到 CSV

    # 绘制对比图
    # 绘制对比图（水平条形图）
    plt.figure(figsize=(2.5, 2))
    y = np.arange(len(method_list))
    palette = sns.color_palette("Paired", len(method_list))  # 可以选择其他调色板，如 "viridis", "coolwarm"
    colors = palette

    plt.barh(y, c_index_list, height=0.8, color=colors, label="C-index")
    # plt.barh(y + 0.2, auc_list, height=0.4, color=colors, label="AUC (5-year survival)")

    plt.yticks(y, method_list)
    plt.ylabel("Methods")
    plt.xlabel("C-index")
    plt.title("Comparison of C-index and 5-year Survival AUC")
    # plt.legend()
    plt.tight_layout()

    # 保存图表
    plt.savefig(path, dpi=1000, bbox_inches='tight')
    # plt.show()


def visualize_BP(go_df):
    # 过滤 BP 通路
    bp_df = go_df[go_df["source"] == "GO:BP"]

    # 选取 top 20 基于 -log10(adjusted p-value)
    top20_bp = bp_df.nlargest(10, "negative_log10_of_adjusted_p_value")

    # 绘制柱状图
    plt.figure(figsize=(2, 2))
    sns.barplot(
        y=top20_bp["term_name"],
        x=top20_bp["negative_log10_of_adjusted_p_value"],
        palette=sns.color_palette("light:royalblue_r", as_cmap=False, n_colors=10)  # 皇家蓝渐变
    )

    # 添加p-value = 0.05的红色虚线
    plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=0.5)
    # 在虚线旁添加文本标注
    # plt.text(-np.log10(0.05) + 0.1, 9, 'p=0.05', color='red', fontsize=6, rotation=90)

    plt.xlabel("-log10 Adjusted p-value")
    plt.ylabel("Biological Process(BP)")
    plt.title("Top 20 Enriched BP")
    # plt.gca().invert_yaxis()  # 让最显著的通路在顶部（已注释）
    plt.savefig('BP_pathway.pdf', dpi=1000, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存


def visualize_immune(immune_df, variable, age_type, path, target_group="Immunologically Quiet (Immune C5)", figsize=(3, 2)):
    # 设置图形风格
    # sns.set_style("whitegrid")

    # 创建分布图
    plt.figure(figsize=figsize)
    sns.violinplot(data=immune_df,
                   x=variable,
                   y=age_type,
                   palette="muted",
                   linewidth=0.5)  # 使用柔和的颜色方案

    # 添加标题和标签
    plt.title(f'Distribution of PAAG by {variable}')
    plt.xlabel('Immune Subtype')
    plt.ylabel('PAAG')

    # 提取目标组数据
    target_data = immune_df[immune_df[variable] == target_group][age_type].dropna()

    # 提取其他所有组的数据并合并
    other_data = immune_df[immune_df[variable] != target_group][age_type].dropna()

    # 计算Mann-Whitney U检验的p值
    p_value = None
    if len(target_data) > 0 and len(other_data) > 0:
        stat, p_value = mannwhitneyu(target_data, other_data, alternative='two-sided')

        # 在图上添加p值
        plt.text(0.5, 0.95, f'p-value (C5 vs others): {p_value:.2e}',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=plt.gca().transAxes,
                 fontsize=6)

    # 调整布局
    plt.tight_layout()

    # 保存图形
    plt.savefig(path, dpi=1000, bbox_inches='tight')
    plt.close()


def visualize_immune_and_immune(immune_df, variable, path):
    from scipy.stats import kruskal, mannwhitneyu
    import itertools
    # 根据 loess_age_gap 划分三组
    immune_df.loc[immune_df['loess_age_gap'] <= -5, 'group'] = 0  # Decelerated
    immune_df.loc[(immune_df['loess_age_gap'] > -5) & (immune_df['loess_age_gap'] <= 5), 'group'] = 1  # Stable
    immune_df.loc[immune_df['loess_age_gap'] > 5, 'group'] = 2 # Accelerated

    group_labels = {0: 'Decelerated', 1: 'Normal', 2: 'Accelerated'}
    immune_df['group_label'] = immune_df['group'].map(group_labels)

    # 创建小提琴图
    plt.figure(figsize=(2, 2))
    sns.boxplot(data=immune_df,
                   x='group_label',  # x轴使用分组标签
                   y=variable,  # y轴使用variable
                   palette="muted",  # 使用柔和的颜色方案
                   width=0.8,  # 设置小提琴宽度
                   order=['Decelerated', 'Normal', 'Accelerated'],
                   linewidth=0.8)  # 固定顺序

    # 添加标题和标签
    plt.title(f'Distribution of {variable} by Aging Groups')
    plt.xlabel('Aging Group')
    plt.ylabel(variable)

    # 调整布局
    plt.tight_layout()

    # 保存图形
    plt.savefig(path, dpi=1000, bbox_inches='tight')
    plt.close()

    # 统计分析
    # 按组别提取variable数据
    grouped_data = [
        immune_df[immune_df['group_label'] == 'Decelerated'][variable].dropna(),
        immune_df[immune_df['group_label'] == 'Normal'][variable].dropna(),
        immune_df[immune_df['group_label'] == 'Accelerated'][variable].dropna()
    ]

    # Kruskal-Wallis 检验（总体差异）
    stat, p_value = kruskal(*grouped_data)
    print(f'Kruskal-Wallis test across all groups:')
    print(f'Statistic = {stat:.4f}, P-value = {p_value}')




# survival_analysis(df)

df = pd.read_csv(f'../../train/agingFound_Age/test_age_multiomics_{tissue}_phenotype_loess_age.csv')
df.dropna(inplace=True)
df1 = pd.read_csv(f'../../train/agingFound_Age/test_BiTage_multiomics_{tissue}_phenotype_loess_age.csv')
df2 = pd.read_csv(f'../../train/agingFound_Age/test_DeepMage_multiomics_{tissue}_phenotype_loess_age.csv')
go_df = pd.read_csv('gProfiler_hsapiens_2025-2-13 17-04-54__intersections.csv')


# savePath = './method_HR.pdf'
survival_analysis(df, 'pancancer_survival.pdf')
Method_HR_analysis([df, df, df1, df1, df2, df2],
                   ['Ours(PAAG)', 'Ours(AG)', 'BitAge(PAAG)', 'BitAge(AG)', 'DeepMAge(PAAG)', 'DeepMAge(AG)'],
                   ['loess_age_gap', 'age_gap', 'loess_age_gap', 'age_gap', 'loess_age_gap', 'age_gap'],
                   'Pancancer_Method_HR.pdf')
Pancancer_HR_analysis(df, ['LGG', 'LUAD', 'LIHC', 'BRCA', 'KIRC', 'COADREAD', 'PAAD'], 'ours_cancer_type_HR.pdf')
Method_c_index_analysis([df, df, df1, df1, df2, df2],
                        ['Ours(PAAG)', 'Ours(AG)', 'BitAge(PAAG)', 'BitAge(AG)', 'DeepMAge(PAAG)', 'DeepMAge(AG)'],
                        ['loess_age_gap', 'age_gap', 'loess_age_gap', 'age_gap', 'loess_age_gap', 'age_gap'],
                        'Pancancer_Method_c_index.pdf')
visualize_BP(go_df)


immune_data = pd.read_csv('/home/wfa/project/multiomics_aging/data/TCGA_tumor_data/clinical_info/Subtype_Immune_Model_Based.txt', sep='\t')
immune_data['sample'] = immune_data['sample'].str[:-3]
immune_data['sample'].drop_duplicates(inplace=True)
immune_data.rename(columns={'sample': 'SampleID'}, inplace=True)

merge_df = df.merge(immune_data, on='SampleID', how='left')


immune_data_cell = pd.read_csv('/home/wfa/project/multiomics_aging/data/TCGA_tumor_data/clinical_info/TCGA_pancancer_10852whitelistsamples_68ImmuneSigs.xena', sep='\t')
immune_data_cell = immune_data_cell.T
immune_data_cell.reset_index(inplace=True)
immune_data_cell.columns = immune_data_cell.iloc[0, :].values
immune_data_cell.drop(0, inplace=True)
immune_data_cell.rename(columns={'Unnamed: 0': 'SampleID'}, inplace=True)
immune_data_cell['SampleID'] = immune_data_cell['SampleID'].str[:-3]

merge_df = merge_df.merge(immune_data_cell, on='SampleID', how='left')
merge_df.to_csv('merge_df.csv', index=False)

print(merge_df.columns)

visualize_immune(merge_df, 'Subtype_Immune_Model_Based', 'loess_age_gap', 'subtype_PAAG.pdf')
visualize_immune_and_immune(merge_df, 'ICS5_score', 'ICS5_score.pdf')
