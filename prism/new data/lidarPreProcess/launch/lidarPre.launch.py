import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # 1. Voxel Downsampling
        Node(
            package='lidarPreProcess',
            executable='lidar_voxel_node',
            name='lidar_voxel_node',
            output='screen',
            parameters=[{'leaf_size': 0.1}]
        ),

        # 2. Statistical Outlier Removal
        Node(
            package='lidarPreProcess',
            executable='lidar_sor_node',
            name='lidar_sor_node',
            output='screen',
            parameters=[
                {'mean_k': 20},
                {'stddev_mul_thresh': 2.0}
            ]
        ),

        Node(
            package='lidarPreProcess',
            executable='lidar_ground_remove_node',
            name='lidar_ground_remove_node',
            output='screen',
            parameters=[
                {'axis_nx': 0.0},
                {'axis_ny': 0.0},
                {'axis_nz': 1.0},
                {'grid_res': 0.5},
                {'height_margin': 0.25},
                {'max_range': 40.0},
                {'dilation_steps': 3},
            ]
        ),

        Node(
            package='lidarPreProcess',
            executable='bounding_box_node',
            name='bounding_box_node',
            output='screen',
            parameters=[
                # ---- Clustering (range-adaptive DBSCAN) ----
                # eps grows linearly with range so far humans (sparse pts) can still be found
                {'cluster_tolerance': 0.20},   # eps at r=0 m (must be >= leaf_size)
                {'eps_range_factor':  0.06},   # eps grows 6%/m → 0.20*(1+0.06*20)=0.44m at 20m
                {'dbscan_min_pts':    5},       # core-point threshold near
                {'min_pts_far':       2},       # core-point threshold at far_range
                {'far_range':        20.0},     # metres at which min_pts_far applies
                {'min_cluster_size':  20},      # min pts per cluster near
                {'min_cluster_size_far': 4},    # min pts per cluster far (human = few pts)
                {'far_cluster_range': 20.0},    # metres at which min_cluster_size_far applies
                {'max_cluster_size':  500000},

                # ---- Z passthrough ----
                {'z_min': -10.0},
                {'z_max':  50.0},

                # ---- Per-cluster filters ----
                {'max_box_dimension':   15.0},
                {'min_box_volume':       0.01},
                {'max_ground_clearance': 0.8}, # slightly relaxed for tilted lidar
                {'min_point_density':    5.0},  # pts/m³ near
                {'density_range_factor': 0.08}, # threshold halves at ~12m, quarters at ~25m
                {'merge_distance':       1.5},

                # ---- Visualisation ----
                {'marker_alpha': 0.60},
                {'skip_frames':  2},
            ]
        ),
    ])
