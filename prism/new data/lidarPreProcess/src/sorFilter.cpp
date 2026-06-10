#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

class LidarSorFilter : public rclcpp::Node {
public:
    using PointCloud2Sub = rclcpp::Subscription<sensor_msgs::msg::PointCloud2>;
    using PointCloud2Pub = rclcpp::Publisher<sensor_msgs::msg::PointCloud2>;

    LidarSorFilter() : Node("lidar_sor_filter")
    {

        this->declare_parameter<int>("mean_k", 20);
        this->declare_parameter<double>("stddev_mul_thresh", 2.0);

        mean_k_           = this->get_parameter("mean_k").as_int();
        stddev_mul_thresh_ = this->get_parameter("stddev_mul_thresh").as_double();

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "velodyne_points_voxel_downsampled", 10,
            std::bind(&LidarSorFilter::topic_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "velodyne_points_sor_filtered", 10);

        RCLCPP_INFO(this->get_logger(),
            "LidarSorFilter started. mean_k=%d  stddev_mul_thresh=%.2f",
            mean_k_, stddev_mul_thresh_);
    }

private:
    void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(mean_k_);
        sor.setStddevMulThresh(stddev_mul_thresh_);
        sor.filter(*cloud_filtered);

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*cloud_filtered, output);

        output.header = msg->header;

        publisher_->publish(output);
    }

    int    mean_k_;
    double stddev_mul_thresh_;

    PointCloud2Sub::SharedPtr subscription_;
    PointCloud2Pub::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarSorFilter>());
    rclcpp::shutdown();
    return 0;
}
