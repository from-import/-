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

# 将点云配准后可视化
print("RANSAC Transformation:", ransac_result.transformation)
print("ICP Transformation:", icp_result.transformation)

# 将源点云转换为配准后的形状
source.transform(icp_result.transformation)
o3d.visualization.draw_geometries([source, target], window_name="Point Cloud Registration Result")

# 特征提取和可视化
o3d.visualization.draw_geometries([source, target], point_show_normal=True, window_name="Feature Visualization")
