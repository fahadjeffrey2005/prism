cmake_minimum_required(VERSION 3.8)
project(lidarPreProcess)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters segmentation sample_consensus search)

add_executable(lidar_voxel_node src/lidarVoxel.cpp)

target_include_directories(lidar_voxel_node PUBLIC
  ${PCL_INCLUDE_DIRS}
)

ament_target_dependencies(lidar_voxel_node
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_msgs
)

target_link_libraries(lidar_voxel_node
  ${PCL_LIBRARIES}
)

install(TARGETS lidar_voxel_node
  DESTINATION lib/${PROJECT_NAME}
)

# SOR filter node
# add_executable(lidar_sor_node src/sorFilter.cpp)

# target_include_directories(lidar_sor_node PUBLIC
#   ${PCL_INCLUDE_DIRS}
# )

# ament_target_dependencies(lidar_sor_node
#   rclcpp
#   sensor_msgs
#   pcl_conversions
# )

# target_link_libraries(lidar_sor_node
#   ${PCL_LIBRARIES}
# )

# install(TARGETS lidar_sor_node
#   DESTINATION lib/${PROJECT_NAME}
# )

# Ground removal node (RANSAC)
add_executable(lidar_ground_remove_node src/lidarGroundRemove.cpp)

target_include_directories(lidar_ground_remove_node PUBLIC
  ${PCL_INCLUDE_DIRS}
)

ament_target_dependencies(lidar_ground_remove_node
  rclcpp
  sensor_msgs
  pcl_conversions
)

target_link_libraries(lidar_ground_remove_node
  ${PCL_LIBRARIES}
)

install(TARGETS lidar_ground_remove_node
  DESTINATION lib/${PROJECT_NAME}
)

# Euclidean Clustering node
add_executable(euclidean_cluster_node src/EuclideanClusters.cpp)

target_include_directories(euclidean_cluster_node PUBLIC
  ${PCL_INCLUDE_DIRS}
)

ament_target_dependencies(euclidean_cluster_node
  rclcpp
  sensor_msgs
  pcl_conversions
)

target_link_libraries(euclidean_cluster_node
  ${PCL_LIBRARIES}
)

install(TARGETS euclidean_cluster_node
  DESTINATION lib/${PROJECT_NAME}
)

# Bounding Box node (clustering + AABB markers for RViz2)
add_executable(bounding_box_node src/boundingBox.cpp)

target_include_directories(bounding_box_node PUBLIC
  ${PCL_INCLUDE_DIRS}
)

ament_target_dependencies(bounding_box_node
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
)

target_link_libraries(bounding_box_node
  ${PCL_LIBRARIES}
)

install(TARGETS bounding_box_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
