import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.loss_function import Reconstruction_loss, KL_loss
import random


def reparameterize(mean, logvar):
    std = torch.exp(logvar / 2)
    epsilon = torch.randn_like(std).cuda()
    return epsilon * std + mean


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def un_dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        un_dfs_freeze(child)


def product_of_experts(mu_set_, log_var_set_):
    tmp = 0
    for i in range(len(mu_set_)):
        tmp += torch.div(1, torch.exp(log_var_set_[i]))

    poe_var = torch.div(1., tmp)
    poe_log_var = torch.log(poe_var)

    tmp = 0.
    for i in range(len(mu_set_)):
        tmp += torch.div(1., torch.exp(log_var_set_[i])) * mu_set_[i]
    poe_mu = poe_var * tmp
    return poe_mu, poe_log_var


class LinearLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout=0.3,
                 batchnorm=False,
                 activation=None):
        super(LinearLayer, self).__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.batchnorm = nn.BatchNorm1d(output_dim) if batchnorm else None

        self.activation = None
        if activation is not None:
            if activation == 'relu':
                self.activation = F.relu
            elif activation == 'sigmoid':
                self.activation = torch.sigmoid
            elif activation == 'tanh':
                self.activation = torch.tanh
            elif activation == 'leakyrelu':
                self.activation = torch.nn.LeakyReLU()
            elif activation == 'softmax':
                self.activation = torch.nn.Softmax()
            elif activation == 'elu':
                self.activation = torch.nn.ELU()
            elif activation == 'selu':
                self.activation = torch.nn.SELU()
            # You can add more activation functions here

    def forward(self, input_x):
        x = self.linear_layer(input_x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ModalVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(ModalVAEEncoder, self).__init__()
        self.FeatureEncoder = nn.ModuleList(
            [LinearLayer(input_dim, hidden_dims[0], batchnorm=True, activation='relu')])
        for i in range(len(hidden_dims) - 1):
            self.FeatureEncoder.append(
                LinearLayer(hidden_dims[i], hidden_dims[i + 1], batchnorm=True, activation='relu'))
        self.mu_predictor = LinearLayer(hidden_dims[-1], latent_dim, batchnorm=True)
        self.logVar_predictor = LinearLayer(hidden_dims[-1], latent_dim, batchnorm=True)

    def forward(self, input_x):
        for layer in self.FeatureEncoder:
            input_x = layer(input_x)
        mu = self.mu_predictor(input_x)
        log_var = self.logVar_predictor(input_x)
        latent_z = reparameterize(mu, log_var)
        return mu, log_var, latent_z

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std).cuda()
        return epsilon * std + mean


class ModalVAEDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(ModalVAEDecoder, self).__init__()
        self.FeatureDecoder = nn.ModuleList(
            [LinearLayer(input_dim, hidden_dims[0], batchnorm=True, activation='relu')])
        for i in range(len(hidden_dims) - 1):
            self.FeatureDecoder.append(
                LinearLayer(hidden_dims[i], hidden_dims[i + 1], batchnorm=True, activation='relu'))
        self.ReconsPredictor = LinearLayer(hidden_dims[-1], output_dim)

    def forward(self, input_x):
        for layer in self.FeatureDecoder:
            input_x = layer(input_x)
        DataRecons = self.ReconsPredictor(input_x)
        return DataRecons


class ModalVAE(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dims, decoder_hidden_dims, latent_dim, input_dist, kl_weight,
                 age_weight):
        super(ModalVAE, self).__init__()
        self.modalVAEEncoder = ModalVAEEncoder(input_dim, encoder_hidden_dims, latent_dim)
        self.modalVAEDecoder = ModalVAEDecoder(latent_dim + 3, decoder_hidden_dims, input_dim)
        self.input_dist = input_dist
        self.kl_weight = kl_weight
        self.age_weight = age_weight

    def forward(self, input_x, dataset_label):
        # input_x_label = torch.concat((input_x, dataset_label, tissue_label), dim=1)
        mu, log_var, latent_z = self.modalVAEEncoder(input_x)

        recon_x = self.modalVAEDecoder(torch.concat((latent_z, dataset_label), dim=1))
        return {'mu': mu, 'log_var': log_var, 'latent_z': latent_z, 'recon_x': recon_x}

    def compute_loss(self, input_x, dataset_label, tissue_label, age_label):
        # input_x_label = torch.concat((input_x), dim=1)
        mu, log_var, latent_z = self.modalVAEEncoder(input_x)
        latent_z = torch.concat((latent_z, dataset_label), dim=1)
        recon_x = self.modalVAEDecoder(latent_z)
        recon_loss = Reconstruction_loss(recon_x, input_x, 1, self.input_dist)
        kl_loss = KL_loss(mu, log_var, 1)

        age_contrastive_loss = self.contrastive_loss(latent_z, age_label)
        return recon_loss + self.kl_weight * kl_loss + self.age_weight * age_contrastive_loss, age_contrastive_loss

    def generate_recon(self, latent_z, dataset_label):
        latent_z = torch.concat((latent_z, dataset_label), dim=1)
        recon_x = self.modalVAEDecoder(latent_z)
        return recon_x

    @staticmethod
    def contrastive_loss(embeddings, labels, margin=0.5, distance='cosine'):

        if distance == 'euclidean':
            distances = torch.cdist(embeddings, embeddings)
        elif distance == 'cosine':
            normed_embeddings = F.normalize(embeddings, p=2, dim=1)
            distances = 1 - torch.mm(normed_embeddings, normed_embeddings.transpose(0, 1))
        else:
            raise ValueError(f"Unknown distance type: {distance}")

        labels_matrix = labels.view(-1, 1) == labels.view(1, -1)

        positive_pair_distances = distances * labels_matrix.float()
        negative_pair_distances = distances * (1 - labels_matrix.float())

        positive_loss = positive_pair_distances.sum() / labels_matrix.float().sum()
        negative_loss = F.relu(margin - negative_pair_distances).sum() / (1 - labels_matrix.float()).sum()

        return positive_loss + negative_loss


class AgingFound(nn.Module):
    def __init__(
            self,
            input_dims,
            encoder_hidden_dims=None,
            decoder_hidden_dims=None,
            latent_dim: int = 128,
            kl_weight: float = 0.001,
            con_weight: float = 10):

        super(AgingFound, self).__init__()
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [1024, 1024]
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [1024, 1024]
        self.kl_weight = kl_weight
        self.con_weight = con_weight

        self.RNAModalVAE = ModalVAE(input_dims[0], encoder_hidden_dims, decoder_hidden_dims, latent_dim,
                                    'gaussian', self.kl_weight, self.con_weight)
        self.MethylationModalVAE = ModalVAE(input_dims[1], encoder_hidden_dims, decoder_hidden_dims, latent_dim,
                                            'gaussian', self.kl_weight, self.con_weight)
        self.proxies = nn.Parameter(torch.randn(9, latent_dim))

    def forward(self, input_x, dataset_label, tissue_label, omics_label):
        if (omics_label[0]) and (not omics_label[1]):
            rna_output = self.RNAModalVAE(input_x[0], dataset_label)
        else:
            rna_output = 0

        if (omics_label[1]) and (not omics_label[0]):
            methylation_output = self.MethylationModalVAE(input_x[1], dataset_label)
        else:
            methylation_output = 0
        # print(methylation_loss)
        if (omics_label[0]) and (omics_label[1]):
            rna_output = self.RNAModalVAE(input_x[0], dataset_label)
            methylation_output = self.MethylationModalVAE(input_x[1], dataset_label)
            poe_mu, poe_log_var = product_of_experts([rna_output['mu'], methylation_output['mu']],
                                                     [rna_output['log_var'], methylation_output['log_var']])
            multiomics_latent_z = reparameterize(poe_mu, poe_log_var)
            multiomics_output = {'mu': poe_mu, 'log_var': poe_log_var, 'latent_z': multiomics_latent_z}
        else:
            multiomics_output = 0
        output = {'rna_output': rna_output, 'methylation_output': methylation_output,
                  'multiomics_output': multiomics_output}

        return output

    def compute_loss(self, input_x, dataset_label, tissue_label, omics_label, age_label):
        if (omics_label[0]) and (not omics_label[1]):
            rna_output = self.RNAModalVAE(input_x[0], dataset_label)
            rna_recon = rna_output['recon_x']
            recon_loss = Reconstruction_loss(rna_recon, input_x[0], 1, 'gaussian')
            kl_loss = KL_loss(rna_output['mu'], rna_output['log_var'], 1)
            age_contrastive_loss = self.ordcon_loss(rna_output['latent_z'], age_label)
            rna_loss = recon_loss + self.kl_weight * kl_loss + self.con_weight * age_contrastive_loss
        else:
            rna_loss = 0

        if (omics_label[1]) and (not omics_label[0]):
            methylation_loss = self.MethylationModalVAE.compute_loss(input_x[1], dataset_label, tissue_label, age_label)
        else:
            methylation_loss = 0

        if (omics_label[0]) and (omics_label[1]):
            rna_output = self.RNAModalVAE(input_x[0], dataset_label)
            methylation_output = self.MethylationModalVAE(input_x[1], dataset_label)
            poe_mu, poe_log_var = product_of_experts([rna_output['mu'], methylation_output['mu']],
                                                     [rna_output['log_var'], methylation_output['log_var']])
            multi_latent_z = reparameterize(poe_mu, poe_log_var)

            kl_loss = KL_loss(poe_mu, poe_log_var, 1.0)

            rna_recon = self.RNAModalVAE.generate_recon(multi_latent_z, dataset_label)
            methylation_recon = self.MethylationModalVAE.generate_recon(multi_latent_z, dataset_label)

            rna_recon_loss = Reconstruction_loss(rna_recon, input_x[0], 1.0, 'gaussian')
            methylation_recon_loss = Reconstruction_loss(methylation_recon, input_x[1], 1.0, 'gaussian')

            age_contrastive_loss = self.ordcon_loss(multi_latent_z, age_label)
            multiomics_loss = (
                                          rna_recon_loss + methylation_recon_loss) + self.kl_weight * kl_loss + self.con_weight * age_contrastive_loss
        else:
            multiomics_loss = 0

        total_loss = rna_loss + methylation_loss + multiomics_loss
        return total_loss, age_contrastive_loss

    @staticmethod
    def contrastive_loss(embeddings, labels, margin=1, distance='cosine'):
        if distance == 'euclidean':
            distances = torch.cdist(embeddings, embeddings)
        elif distance == 'cosine':
            normed_embeddings = F.normalize(embeddings, p=2, dim=1)
            distances = 1 - torch.mm(normed_embeddings, normed_embeddings.transpose(0, 1))
        else:
            raise ValueError(f"Unknown distance type: {distance}")

        labels_matrix = labels.view(-1, 1) == labels.view(1, -1)

        # 正样本损失：相同标签的距离均值
        positive_mask = labels_matrix.fill_diagonal_(False)  # 排除自身
        positive_distances = distances[positive_mask]
        positive_loss = positive_distances.mean() if len(positive_distances) > 0 else 0.0

        # 负样本损失：不同标签的距离小于margin时的均值
        negative_mask = ~labels_matrix
        negative_distances = distances[negative_mask]
        negative_loss = F.relu(margin - negative_distances).mean() if len(negative_distances) > 0 else 0.0

        return positive_loss + negative_loss

    def ordcon_loss(self, embeddings, ages, temperature=0.9):
        """
        修正后的损失函数，包含软代理匹配和正确对比损失结构
        """
        N = embeddings.size(0)
        num_proxies = self.proxies.size(0)

        # 1. 计算所有样本与所有代理的余弦相似度 [N, num_proxies]
        sim_matrix = torch.exp(
            F.cosine_similarity(embeddings.unsqueeze(1), self.proxies.unsqueeze(0), dim=-1) / temperature)

        # 2. 获取正样本索引（假设年龄从0开始）
        positive_indices = ages.long() - 1  # 若年龄从1开始需-1
        positive_indices = torch.squeeze(positive_indices, dim=1)

        # 3. 计算动态权重矩阵 [N, num_proxies]
        age_diff = torch.abs(ages.view(-1, 1) - torch.arange(num_proxies, device=ages.device).float())
        weights = 1 / (1 + torch.exp(-age_diff / age_diff.max(dim=1, keepdim=True).values))
        # # 行索引
        rows = torch.arange(N)
        # # 确保权重矩阵正确置为 0，仅对应正样本索引
        weights[rows, positive_indices] = 0

        # 4. 计算分子（正样本相似度）和分母（加权负样本相似度）
        sim_pos = sim_matrix[torch.arange(N), positive_indices]
        sim_neg = (sim_matrix * weights).sum(dim=1)
        # 5. 计算损失（防止数值溢出）
        loss = - (sim_pos / sim_neg).mean()
        #
        # # ================== 新增方向向量对比损失 ==================
        # # 1. 计算所有样本对方向向量 [N, N, feat_dim]
        # emb_diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)  # 特征差
        # dir_vectors = F.normalize(emb_diff, p=2, dim=-1)  # 单位化方向向量
        #
        # forward_embeddings = self.proxies[positive_indices]
        # forward_diff = forward_embeddings.unsqueeze(1) - forward_embeddings.unsqueeze(0)
        # forward_vectors = F.normalize(forward_diff, p=2, dim=-1)
        #
        # result = torch.exp(torch.einsum('ijk,ijk->ij', dir_vectors, forward_vectors))  # [N, N]
        # result = result / temperature
        #
        # forward_matrix = (ages.T > ages).int()  # [N, N]
        #
        # q = torch.sum(result * forward_matrix, dim=1)
        # q_ = torch.sum(result, dim=1)
        # q_forward = torch.sum(q / q_, dim=0)

        return loss


class MultiAge(nn.Module):
    def __init__(self, input_dim, latent_dim, model_path):
        super(MultiAge, self).__init__()
        self.agingFound = AgingFound(input_dim)
        if model_path != '':
            modal_dict = torch.load(model_path, map_location='cpu')
            self.agingFound.load_state_dict(modal_dict)
        # dfs_freeze(self.agingFound)

        self.agePredictor = nn.Sequential(LinearLayer(latent_dim, 128, dropout=0.3, batchnorm=True, activation='relu'), )

        self.outLayer = LinearLayer(128, 100, dropout=0.3)

    def latent_embedding(self, input_x, dataset_label, tissue_label):
        output = self.modalVAE(input_x, dataset_label, tissue_label)
        return output

    def forward(self, input_x, dataset_label, tissue_label, omics_label):
        latent_output = self.agingFound(input_x, dataset_label, tissue_label, omics_label)

        if (omics_label[0]) and (not omics_label[1]):
            output = latent_output['rna_output']

        if (omics_label[1]) and (not omics_label[0]):
            output = latent_output['methylation_output']

        if (omics_label[0]) and (omics_label[1]):
            output = latent_output['multiomics_output']

        latent_mu = output['mu']
        agePredict = self.agePredictor(latent_mu)
        agePredict_logit = self.outLayer(agePredict)

        return agePredict_logit

    def age_predict(self, input_x, dataset_label, tissue_label, omics_label):
        agePredictLogit = self.forward(input_x, dataset_label, tissue_label, omics_label)
        agePredictLogit = F.softmax(agePredictLogit, dim=1)

        bins = torch.arange(1, 101).float().cuda()  # shape = [100]

        # 4. 对每个样本，计算期望值(加权求和)得到最终预测年龄
        #    predicted_age.shape = [batch_size]
        predicted_age = torch.sum(agePredictLogit * bins, dim=1).unsqueeze(1)

        return predicted_age


class OrderContrastiveLoss(nn.Module):
    def __init__(self, num_ages, feat_dim, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        # 初始化每个年龄的代理（Proxy）
        self.proxies = nn.Parameter(torch.randn(num_ages, feat_dim))

    def forward(self, z_age, ages):
        """
        z_age: 提取的年龄特征 [B, D]
        ages: 真实年龄标签 [B]
        """
        loss = 0
        batch_size = z_age.size(0)

        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    continue

                age_i = ages[i].item()
                age_j = ages[j].item()

                # 计算特征方向向量
                z_i, z_j = z_age[i], z_age[j]
                v_d = (z_i - z_j) / (torch.norm(z_i - z_j) + 1e-6)  # Eq(1)

                # 根据年龄关系选择参考方向
                if age_i < age_j:
                    # 递增关系：对比正向和反向参考方向
                    c_current = self.proxies[int(age_j)]
                    c_next = self.proxies[min(int(age_j) + 1, len(self.proxies) - 1)]
                    c_prev = self.proxies[max(int(age_j) - 1, 0)]

                    v_f = (c_next - c_current) / (torch.norm(c_next - c_current) + 1e-6)  # Eq(2)
                    v_b = (c_prev - c_current) / (torch.norm(c_prev - c_current) + 1e-6)  # Eq(3)

                    # 计算相似度得分
                    sim_pos = torch.exp(torch.dot(v_d, v_f) / self.temperature)
                    sim_neg = torch.exp(torch.dot(v_d, v_b) / self.temperature)

                    loss += -torch.log(sim_pos / (sim_pos + sim_neg))  # Eq(4)

                elif age_i > age_j:
                    # 递减关系（类似逻辑，方向反向）
                    ...

        return loss / (batch_size * (batch_size - 1))


class ExplainMultiAge(nn.Module):
    def __init__(self, input_dim, latent_dim, model_path, omics_label,
                 split_size_section=None):
        super(ExplainMultiAge, self).__init__()
        if split_size_section is None:
            self.split_size_section = [11816, 6617, 3, 64]
        else:
            self.split_size_section = split_size_section
        self.multiAge = MultiAge(input_dim, latent_dim, '')
        if model_path is not None:
            modal_dict = torch.load(model_path, map_location='cpu')
            self.multiAge.load_state_dict(modal_dict)
        self.omics_label = omics_label

    def forward(self, input_x):
        rna_input, atac_input, dataset_id, tissue_label = torch.split(input_x, self.split_size_section, dim=1)
        omics_output = self.multiAge.age_predict([rna_input, atac_input], dataset_id, tissue_label, self.omics_label)

        return omics_output


class DNNAgePredictor(nn.Module):
    def __init__(self, input_dim):
        super(DNNAgePredictor, self).__init__()
        self.RNAEncoder = nn.Sequential(LinearLayer(input_dim[0], 1024),
                                        LinearLayer(1024, 64))
        self.MethylationEncoder = nn.Sequential(LinearLayer(input_dim[1], 1024),
                                                LinearLayer(1024, 64))
        self.agePredictor = nn.Sequential(LinearLayer(64, 32), )
        self.outLayer = LinearLayer(32, 1)

    def forward(self, input_x, dataset_label, tissue_label, omics_label):
        rnaEmbedding = self.RNAEncoder(input_x[0])
        methylationEmbedding = self.MethylationEncoder(input_x[1])
        multiembedding = torch.concat((rnaEmbedding, methylationEmbedding), dim=1)
        ageEmbedding = self.agePredictor(rnaEmbedding)
        predict_age = self.outLayer(ageEmbedding)
        return predict_age


class BiTagePredictor(nn.Module):
    def __init__(self, input_dim):
        super(BiTagePredictor, self).__init__()
        self.RNAEncoder = nn.Linear(input_dim[0], 1)
        self.MethylationEncoder = nn.Linear(input_dim[1], 1)

    def forward(self, input_x):
        predict_age = self.RNAEncoder(input_x[0])
        return predict_age


class DeepMAgePredictor(nn.Module):
    def __init__(self, input_dims):
        super(DeepMAgePredictor, self).__init__()
        self.omics_encoder = nn.Sequential(LinearLayer(input_dims[0], 512, activation='elu'),
                                           LinearLayer(512, 512, activation='elu'),
                                           LinearLayer(512, 512, activation='elu'),
                                           LinearLayer(512, 1))

    def forward(self, input_x):
        # input_x = torch.concat(input_x, dim=1)
        omics_output = self.omics_encoder(input_x[0])
        return omics_output
