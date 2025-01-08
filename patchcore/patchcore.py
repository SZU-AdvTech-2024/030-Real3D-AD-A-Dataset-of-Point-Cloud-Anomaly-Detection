"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
# import patchcore.common_pointCore
import patchcore.sampler

from timm.models import create_model
import argparse
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from sklearn.cluster import KMeans
import open3d as o3d
from M3DM.cpu_knn import fill_missing_values
from feature_extractors.ransac_position import get_registration_np,get_registration_refine_np
from utils.utils import get_args_point_mae
from M3DM.models import Model1
from sklearn.decomposition import PCA

from scipy.spatial import distance_matrix
from scipy.sparse.linalg import eigsh
from pointnet2_ops import pointnet2_utils
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from multiprocessing import Pool
LOGGER = logging.getLogger(__name__)

def adaptive_gaussian_weight(distances, k):
    """
    根据点的邻域动态调整高斯核宽度
    :param distances: 当前点到其他点的距离数组
    :param k: 最近邻数量
    :return: 动态高斯核权重
    """
    sigma = np.mean(distances[:k])  # 基于 k 最近邻的距离均值调整
    weights = np.exp(-distances**2 / (2 * sigma**2))
    return weights

def compute_chunk_weights(chunk_indices, points, k):
    """
    并行计算一个分块中的权重矩阵
    :param chunk_indices: 当前分块中的点索引
    :param points: 点云数据，形状为 (N, D)
    :param k: 最近邻数量
    :return: 权重的 (rows, cols, values) 三元组
    """
    tree = KDTree(points)  # 构建 KD 树
    rows, cols, weights = [], [], []
        
    for i in chunk_indices:
        distances, indices = tree.query(points[i], k=k + 1)  # 查找 k+1 邻居
        dynamic_weights = adaptive_gaussian_weight(distances[1:], k)  # 自适应高斯核
        for j, weight in zip(indices[1:], dynamic_weights):  # 跳过自身点
            rows.append(i)
            cols.append(j)
            weights.append(weight)
        
    return rows, cols, weights

def build_laplacian_large_scale(points, k=10, chunk_size=1000, normalized=False, n_jobs=4):
    """
    构建大规模点云的稀疏图拉普拉斯矩阵
    :param points: 点云数据，形状为 (N, D)
    :param k: 每个点的 k 近邻
    :param chunk_size: 每块点云的大小
    :param normalized: 是否生成归一化拉普拉斯矩阵
    :param n_jobs: 并行处理的进程数
    :return: 稀疏图拉普拉斯矩阵
    """
    N = points.shape[0]
    indices = np.arange(N)
    chunks = [indices[i:i + chunk_size] for i in range(0, N, chunk_size)]  # 分块索引

    # 并行计算每块的权重
    with Pool(n_jobs) as pool:
        results = pool.starmap(compute_chunk_weights, [(chunk, points, k) for chunk in chunks])

    # 合并所有块的 (rows, cols, weights)
    rows, cols, weights = [], [], []
    for r, c, w in results:
        rows.extend(r)
        cols.extend(c)
        weights.extend(w)
        
    # 构建稀疏邻接矩阵
    W = csr_matrix((weights, (rows, cols)), shape=(N, N))
    D = csr_matrix(np.diag(W.sum(axis=1).A.flatten()))  # 度矩阵
        
    # 构建拉普拉斯矩阵
    if normalized:
        D_inv_sqrt = csr_matrix(np.diag(1.0 / np.sqrt(W.sum(axis=1).A.flatten() + 1e-12)))
        L = csr_matrix(np.eye(N)) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        L = D - W

    return L

class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        basic_template=None,
        num_samples=9400,
        **kwargs,
        
    ):
        # self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})
        self.voxel_size = 0.5 #0.1

        self.num_samples = num_samples
        # feature_aggregator = patchcore.common.NetworkFeatureAggregator(
        #     self.backbone, self.layers_to_extract_from, self.device
        # )
        # feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        # self.forward_modules["feature_aggregator"] = feature_aggregator

        # preprocessing = patchcore.common.Preprocessing(
        #     feature_dimensions, pretrain_embed_dimension
        # )
        # self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        # self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
        #     device=self.device, target_size=input_shape[-2:]
        # )

        self.featuresampler = featuresampler
        self.dataloader_count = 0
        self.basic_template = basic_template
        self.deep_feature_extractor = None
        # self.pca = PCA(n_components=10)
        
    def set_deep_feature_extractor(self):
        # args = get_args_point_mae()
        self.deep_feature_extractor = Model1(device='cuda', 
                        rgb_backbone_name='vit_base_patch8_224_dino', 
                        xyz_backbone_name='Point_MAE', 
                        group_size = 128, 
                        num_group = 16384)
        self.deep_feature_extractor = self.deep_feature_extractor.cuda()
    
    def set_dataloadercount(self, dataloader_count):
        self.dataloader_count = dataloader_count

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)
    
    def embed_xyz(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed_xyz(data)
    
    def _embed_xyz(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        reg_data = reg_data.astype(np.float32)
        return reg_data
    
    def _embed_fpfh(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reg_data))
        radius_normal = self.voxel_size * 2
        o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
        (radius=radius_feature, max_nn=100))
        fpfh = pcd_fpfh.data.T
        # fpfh = torch.from_numpy(pcd_fpfh.data.T)
        # print(fpfh.shape)
        fpfh = fpfh.astype(np.float32)
        return fpfh
    
    def _embed_pointmae(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        pointcloud_data = torch.from_numpy(reg_data).permute(1,0).unsqueeze(0).cuda().float()
        pmae_features, center, ori_idx, center_idx = self.deep_feature_extractor(pointcloud_data)
        pmae_features = pmae_features.squeeze(0).permute(1,0).cpu().numpy()
        print(f"pmae_features:{pmae_features.shape}")
        print(f"pmae_features:{type(pmae_features)}")
    #     fft_results = []
    #     for i in range(pmae_features.shape[1]):
    # # 选择当前坐标轴的数据
    #         coord = pmae_features[:, i]
    # # 对当前坐标轴的数据进行傅里叶变换，并将其添加到结果列表中
    #         fft_results.append(np.fft.fft(coord))
    #     pmae_features = np.column_stack(fft_results)
        pmae_features = pmae_features.astype(np.float32)
        return pmae_features,center_idx

    def _embed_downpointmae_xyz(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        pointcloud_data = torch.from_numpy(reg_data).permute(1,0).unsqueeze(0).cuda().float()
        pmae_features, center, ori_idx, center_idx = self.deep_feature_extractor(pointcloud_data)
        pmae_features = pmae_features.squeeze(0).permute(1,0).cpu().numpy()
        pmae_features = pmae_features.astype(np.float32)
        # pmae_features = self.pca.fit_transform(pmae_features)
        mask_idx = center_idx.squeeze().long()
        xyz = reg_data[mask_idx.cpu().numpy(),:]
        xyz = xyz.repeat(333,1)
        features = np.concatenate([pmae_features,xyz],axis=1)
        return features.astype(np.float32),center_idx
    def build_graph_laplacian(self, points, sigma=0.1):
        """
        根据点云构建图的拉普拉斯矩阵
        """
        # print(f"points:{points.shape}")
        # 计算权重矩阵 (高斯核)
        dist = distance_matrix(points, points)
        W = np.exp(-dist**2 / (2 * sigma**2))
        np.fill_diagonal(W, 0)  # 对角线权重为0

        # 度矩阵
        D = np.diag(W.sum(axis=1))

        # 拉普拉斯矩阵
        L = D - W
        return L
    def graph_fourier_transform(self, L, points):
        """
        对点云应用图傅里叶变换
        """
        eigvals, eigvecs = eigsh(L, k=points.shape[0], which="SM")  # 保留所有特征值
        # 转换到频域
        points_freq = eigvecs.T @ points
        return points_freq, eigvecs
    def fps_sampling(self, points, num_samples):
        """
        使用 Farthest Point Sampling (FPS) 下采样点云
        """
        n_points = points.shape[0]
        sampled_indices = [np.random.randint(n_points)]  # 随机选择第一个点
        distances = np.full(n_points, np.inf)

        for _ in range(1, num_samples):
            last_sampled = sampled_indices[-1]
            dist_to_last_sampled = np.linalg.norm(points - points[last_sampled], axis=1)
            distances = np.minimum(distances, dist_to_last_sampled)
            next_sampled = np.argmax(distances)
            sampled_indices.append(next_sampled)

        return points[sampled_indices], sampled_indices
    def _embed_fft(self, point_cloud, detach=True):
        # print(f"point_cloud:{type(point_cloud)}")
        # print(f"point_cloud:{point_cloud.shape}")

        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reg_data))
        points = np.asarray(o3d_pc.points)
        fft_x = np.fft.fft(points[:, 0])  # 对X坐标进行傅里叶变换
        fft_y = np.fft.fft(points[:, 1])  # 对Y坐标进行傅里叶变换
        fft_z = np.fft.fft(points[:, 2])  # 对Z坐标进行傅里叶变换
        points_fft = np.column_stack((fft_x, fft_y, fft_z))
        return points_fft.astype(np.float32)
    
    def _embed_freq(self, point_cloud, detach=True):
        # print(f"point_cloud:{type(point_cloud)}")
        # print(f"point_cloud:{point_cloud.shape}")

        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)

        # print(f"reg_data:{type(reg_data)}")
        # print(f"reg_data:{reg_data.shape}")
        pointcloud_data , sampled_indices= self.fps_sampling(reg_data,1000)
        # pointcloud_data = torch.from_numpy(reg_data).permute(1,0).unsqueeze(0).cuda().float()
        L = self.build_graph_laplacian(pointcloud_data)
        print(f"L:{L.shape}")
       

        points_freq, eigvecs = self.graph_fourier_transform(L, pointcloud_data)
        return points_freq.astype(np.float32), torch.tensor(sampled_indices)
    
    def _embed_upfpfh_xyz(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reg_data))
        radius_normal = self.voxel_size * 2
        o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
        (radius=radius_feature, max_nn=100))
        fpfh = pcd_fpfh.data.T
        # fpfh = torch.from_numpy(pcd_fpfh.data.T)
        # print(fpfh.shape)
        fpfh = fpfh.astype(np.float32)
        xyz = reg_data.repeat(11,1)
        features = np.concatenate([fpfh,xyz],axis=1)
        return features.astype(np.float32)
    
    # def _embed(self, images, detach=True, provide_patch_shapes=False):
    #     """Returns feature embeddings for images."""

    #     def _detach(features):
    #         if detach:
    #             return [x.detach().cpu().numpy() for x in features]
    #         return features

    #     _ = self.model.eval()
    #     with torch.no_grad():
    #         features = self.model(images)['seg_feat']
    #         features = features.reshape(-1,768)
    #         # print(features.shape)

    #     patch_shapes = [14,14]
    #     if provide_patch_shapes:
    #         return _detach(features), patch_shapes
    #     return _detach(features)
    

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])
    
    def fit_with_return_feature(self, training_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            training_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])
        return features
    
    def get_all_features(self, training_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            training_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        return features
    
    def fit_with_limit_size(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                
                return self._embed_xyz(input_pointcloud)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features
    
    def fit_with_limit_size_fpfh(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_fpfh(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size_fpfh(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                
                return self._embed_fpfh(input_pointcloud)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features

    def fit_with_limit_size_fpfh_upxyz(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_fpfh_upxyz(training_data, limit_size)
    
    def _fill_memory_bank_with_limit_size_fpfh_upxyz(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                
                return self._embed_upfpfh_xyz(input_pointcloud)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features

    def fit_with_limit_size_pmae(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_pmae(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size_pmae(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                pmae_features, sample_idx =self._embed_pointmae(input_pointcloud)
                return pmae_features

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features

    def fit_with_limit_size_downpmae_xyz(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_downpmae_xyz(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size_downpmae_xyz(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                pmae_features, sample_idx =self._embed_downpointmae_xyz(input_pointcloud)
                return pmae_features

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features
    def fit_with_limit_size_fft(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_fft(training_data, limit_size)

    def _fill_memory_bank_with_limit_size_fft(self, input_data, limit_size):
        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                # print(f"input_pointcloud:{input_pointcloud.shape}")
                # print(f"input_pointcloud:{type(input_pointcloud)}")

                # input_pointcloud , _ = self.fps_sampling(input_pointcloud[0], self.num_samples)
                # input_pointcloud = np.expand_dims(input_pointcloud, axis=0)
                # freq_features = self._embed_freq(torch.from_numpy(input_pointcloud))

                freq_features = self._embed_fft(input_pointcloud)
                return freq_features

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features
    
    def fit_with_limit_size_freq(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_freq(training_data, limit_size)

    def _fill_memory_bank_with_limit_size_freq(self, input_data, limit_size):
        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                # print(f"input_pointcloud:{input_pointcloud.shape}")
                # print(f"input_pointcloud:{type(input_pointcloud)}")

                # input_pointcloud , _ = self.fps_sampling(input_pointcloud[0], self.num_samples)
                # input_pointcloud = np.expand_dims(input_pointcloud, axis=0)
                # freq_features = self._embed_freq(torch.from_numpy(input_pointcloud))

                freq_features , _= self._embed_freq(input_pointcloud)
                return freq_features

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict(input_pointcloud)
                # for score, mask in zip(_scores, _masks):
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        # images = images.to(torch.float).to(self.device)
        # _ = self.forward_modules.eval()

        # batchsize = images.shape[0]
        with torch.no_grad():
            features = self._embed_fft(input_pointcloud)
            # print(patch_shapes) [32,32]
            features = np.asarray(features)
            # print(features.shape)
            # features = np.repeat(features,2,axis=1)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
            # print(patch_scores.shape)
            # print(image_scores)
            # image_scores = self.patch_maker.unpatch_scores(
            #     image_scores, batchsize=batchsize
            # )
            # image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            # print(image_scores.shape)
            # image_scores = self.patch_maker.score(image_scores)
            # print(image_scores.shape)

            # patch_scores = self.patch_maker.unpatch_scores(
            #     patch_scores, batchsize=batchsize
            # )
            # patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            # masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [image_scores], [mask for mask in patch_scores]
        # return [score for score in image_scores], [mask for mask in image_scores]
    
    def predict_fpfh(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_fpfh(data)
        return self._predict_fpfh(data)

    def _predict_dataloader_fpfh(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_fpfh(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict_fpfh(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        with torch.no_grad():
            features = self._embed_fpfh(input_pointcloud)
            features = np.asarray(features)
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
        return [image_scores], [mask for mask in patch_scores]
    
    def predict_fft(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_fft(data)
        return self._predict_fft(data)
    def _predict_dataloader_fft(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict(input_pointcloud)
                # for score, mask in zip(_scores, _masks):
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict_fft(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        # images = images.to(torch.float).to(self.device)
        # _ = self.forward_modules.eval()

        # batchsize = images.shape[0]
        with torch.no_grad():
            features = self._embed_xyz(input_pointcloud)
            # print(patch_shapes) [32,32]
            features = np.asarray(features)
            # print(features.shape)
            # features = np.repeat(features,2,axis=1)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
            # print(patch_scores.shape)
            # print(image_scores)
            # image_scores = self.patch_maker.unpatch_scores(
            #     image_scores, batchsize=batchsize
            # )
            # image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            # print(image_scores.shape)
            # image_scores = self.patch_maker.score(image_scores)
            # print(image_scores.shape)

            # patch_scores = self.patch_maker.unpatch_scores(
            #     patch_scores, batchsize=batchsize
            # )
            # patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            # masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [image_scores], [mask for mask in patch_scores]
        # return [score for score in image_scores], [mask for mask in image_scores]
    

    def predict_freq(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_freq(data)
        return self._predict_freq(data)
    
    def fps_downsample(self, points, mask, num_samples):
        """
        Perform Farthest Point Sampling (FPS) on point cloud data with associated mask.
        
        Parameters:
        - points (np.ndarray): A numpy array of shape (N, D) representing N points of D dimensions.
        - mask (np.ndarray): A numpy array of shape (N,) representing the mask associated with each point.
        - num_samples (int): Number of points to sample.

        Returns:
        - sampled_points (np.ndarray): A numpy array of shape (num_samples, D) with sampled points.
        - sampled_mask (np.ndarray): A numpy array of shape (num_samples,) with the mask for sampled points.
        """
        if num_samples > points.shape[0]:
            raise ValueError("Number of samples cannot exceed the number of points.")

        # Initialize an array to store the indices of sampled points
        sampled_indices = []
        # Start with a random point
        initial_index = np.random.randint(0, points.shape[0])
        sampled_indices.append(initial_index)

        # Calculate distances
        distances = np.full(points.shape[0], np.inf)

        for _ in range(1, num_samples):
            # Update distances with the most recently added point
            last_sampled_point = points[sampled_indices[-1]]
            dist_to_last = np.linalg.norm(points - last_sampled_point, axis=1)
            distances = np.minimum(distances, dist_to_last)

            # Choose the farthest point
            farthest_index = np.argmax(distances)
            sampled_indices.append(farthest_index)

        # Extract the sampled points and their corresponding mask values
        sampled_indices = np.array(sampled_indices)
        sampled_points = points[sampled_indices]
        sampled_mask = mask[sampled_indices]

        return sampled_points, sampled_mask

    def _predict_dataloader_freq(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                # print(f"input_pointcloud:{type(input_pointcloud)}")
                # print(f"input_pointcloud:{input_pointcloud.shape}")

                # print(f"mask:{type(mask)}")
                # print(f"mask:{mask.shape}")

                # input_pointcloud, mask = self.fps_downsample(input_pointcloud[0], mask[0], self.num_samples)
                # input_pointcloud = np.expand_dims(input_pointcloud, axis=0)
                # mask = np.expand_dims(mask, axis=0)

                # input_pointcloud = torch.from_numpy(input_pointcloud)
                # mask = torch.from_numpy(mask)
                
                # print(f"input_pointcloud:{type(input_pointcloud)}")
                # print(f"input_pointcloud:{input_pointcloud.shape}")

                # print(f"mask:{type(mask)}")
                # print(f"mask:{mask.shape}")
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_freq(input_pointcloud)
                # print(f"_masks:{type(_masks)}")
                # print(f"_masks:{len(_masks)}")
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt
    
    def _predict_freq(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        with torch.no_grad():
            print(f"input_pointcloud:{type(input_pointcloud)}")
            print(f"input_pointcloud:{input_pointcloud.shape}")
            features, sample_dix = self._embed_freq(input_pointcloud)
            features = np.asarray(features)
            print(f"features:{type(features)}")
            print(f"features:{features.shape}")
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
            mask_idx = sample_dix.squeeze().long()
            xyz_sampled = input_pointcloud[0][mask_idx.cpu(),:]
            full_scores = fill_missing_values(xyz_sampled,patch_scores,input_pointcloud[0],k=1)
        return [image_scores], [mask for mask in full_scores]
    
    def predict_fpfh_upxyz(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_fpfh_upxyz(data)
        return self._predict_fpfh_upxyz(data)

    def _predict_dataloader_fpfh_upxyz(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_fpfh_upxyz(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict_fpfh_upxyz(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        with torch.no_grad():
            features = self._embed_upfpfh_xyz(input_pointcloud)
            features = np.asarray(features)
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
        return [image_scores], [mask for mask in patch_scores]
    
    def predict_pmae(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_pmae(data)
        return self._predict_pmae(data)

    def _predict_dataloader_pmae(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_pmae(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict_pmae(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        with torch.no_grad():
            features, sample_dix = self._embed_pointmae(input_pointcloud)
            features = np.asarray(features,order='C').astype('float32')
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
            mask_idx = sample_dix.squeeze().long()
            xyz_sampled = input_pointcloud[0][mask_idx.cpu(),:]
            # print(patch_scores.shape)
            # print(input_pointcloud.shape)
            # print(xyz_sampled.shape)
            full_scores = fill_missing_values(xyz_sampled,patch_scores,input_pointcloud[0], k=1)
        return [image_scores], [mask for mask in full_scores]

    
    def predict_downpmae_xyz(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_downpmae_xyz(data)
        return self._predict_downpmae_xyz(data)

    def _predict_dataloader_downpmae_xyz(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_downpmae_xyz(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict_downpmae_xyz(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        with torch.no_grad():
            features, sample_dix = self._embed_downpointmae_xyz(input_pointcloud)
            features = np.asarray(features,order='C').astype('float32')
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
            mask_idx = sample_dix.squeeze().long()
            xyz_sampled = input_pointcloud[0][mask_idx.cpu(),:]
            # print(patch_scores.shape)
            # print(input_pointcloud.shape)
            # print(xyz_sampled.shape)
            full_scores = fill_missing_values(xyz_sampled,patch_scores,input_pointcloud[0], k=1)
        return [image_scores], [mask for mask in full_scores]
    
    def _predict_past_tasks(self, features, data):
        pass
            
    def _fit_past_tasks(self, features, data):
        pass
        

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
