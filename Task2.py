import open3d as o3d
import numpy as np

# 读取点云
source = o3d.io.read_point_cloud("dataCloud2_phone_pos1.ply")
target = o3d.io.read_point_cloud("dataCloud2_phone_pos2.ply")


# 点云预处理（去噪、下采样）
def preprocess_point_cloud(pcd, voxel_size):
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd


voxel_size = 0.05  # 定义体素大小
source_down = preprocess_point_cloud(source, voxel_size)
target_down = preprocess_point_cloud(target, voxel_size)


# 特征提取（FPFH）
def compute_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh


source_fpfh = compute_fpfh(source_down, voxel_size)
target_fpfh = compute_fpfh(target_down, voxel_size)

# 粗配准（RANSAC）
ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_down, target_down, source_fpfh, target_fpfh, mutual_filter=False,
    max_correspondence_distance=voxel_size * 2,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=4,
    checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
              o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 2)],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
)

# 确保 source 和 target 点云有法向量
source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

# 精配准（ICP）使用法向量计算后的点云
icp_result = o3d.pipelines.registration.registration_icp(
    source, target, voxel_size * 0.4, ransac_result.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

# 将源点云转换为配准后的形状
source.transform(icp_result.transformation)


# 特征线提取：基于曲率来提取特征点
def extract_feature_lines(pcd, curvature_threshold=0.1):
    # 计算曲率
    curvatures = []
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(len(pcd.points)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 30)
        neighbors = np.asarray(pcd.points)[idx[1:], :]
        center = np.asarray(pcd.points)[i]
        cov_matrix = np.cov((neighbors - center).T)
        eigenvalues, _ = np.linalg.eig(cov_matrix)
        curvature = eigenvalues.min() / eigenvalues.sum()  # 简单计算曲率
        curvatures.append(curvature)

    # 提取曲率大于阈值的点作为特征线
    feature_points = np.asarray(pcd.points)[np.array(curvatures) > curvature_threshold]
    feature_pcd = o3d.geometry.PointCloud()
    feature_pcd.points = o3d.utility.Vector3dVector(feature_points)
    return feature_pcd


# 从源点云和目标点云中提取特征线
source_feature_lines = extract_feature_lines(source)
target_feature_lines = extract_feature_lines(target)

# 可视化配准结果和特征线
o3d.visualization.draw_geometries([source, target], window_name="Point Cloud Registration Result")
o3d.visualization.draw_geometries([source_feature_lines, target_feature_lines], window_name="Feature Lines")


# 计算配准精度（RMSE）
def calculate_rmse(source, target, transformation):
    source_copy = source.transform(transformation)
    distances = source_copy.compute_point_cloud_distance(target)
    rmse = np.sqrt(np.mean(np.asarray(distances) ** 2))
    return rmse


rmse = calculate_rmse(source, target, icp_result.transformation)
print("配准的RMSE精度指标:", rmse)
