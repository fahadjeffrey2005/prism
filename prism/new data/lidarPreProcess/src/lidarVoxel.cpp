#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

class LidarVoxelDownsample : public rclcpp::Node {
public:
    using PointCloud2Sub = rclcpp::Subscription<sensor_msgs::msg::PointCloud2>;
    using PointCloud2Pub = rclcpp::Publisher<sensor_msgs::msg::PointCloud2>;

    LidarVoxelDownsample() : Node("lidar_voxel_downsample")
    {
        this->declare_parameter<double>("leaf_size", 0.2);
        leaf_size_ = static_cast<float>(this->get_parameter("leaf_size").as_double());

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "velodyne_points", 10,
            std::bind(&LidarVoxelDownsample::topic_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "velodyne_points_voxel_downsampled", 10);

        RCLCPP_INFO(this->get_logger(),
            "LidarVoxelDownsample started. leaf_size=%.3f m", leaf_size_);
    }

private:
    void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty cloud, skipping.");
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
        vg.filter(*cloud_filtered);

        RCLCPP_DEBUG(this->get_logger(),
            "Voxel downsample: %zu -> %zu points", cloud->size(), cloud_filtered->size());

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*cloud_filtered, output);
        output.header = msg->header;   // preserve frame_id and timestamp
        publisher_->publish(output);
    }

    float leaf_size_;
    PointCloud2Sub::SharedPtr subscription_;
    PointCloud2Pub::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarVoxelDownsample>());
    rclcpp::shutdown();
    return 0;
}
