#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <deque>

// Colour palette — cycles through clusters
static const uint8_t PALETTE[][3] = {
    {230,  25,  75},   // red
    { 60, 180,  75},   // green
    { 67, 133, 255},   // blue
    {255, 160,  20},   // orange
    {210, 245,  60},   // lime
    {250, 190, 212},   // pink
    {  0, 220, 220},   // cyan
    {220, 190, 255},   // lavender
};
static constexpr std::size_t N_COLOURS = sizeof(PALETTE) / sizeof(PALETTE[0]);

static constexpr int DBSCAN_UNVISITED = -1;
static constexpr int DBSCAN_NOISE     = -2;

void extractDBSCAN(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
    const pcl::search::KdTree<pcl::PointXYZ>::Ptr & tree,
    float eps,
    int   minPts,
    int   min_cluster_size,
    int   max_cluster_size,
    std::vector<pcl::PointIndices> & clusters)
{
    const int n = static_cast<int>(cloud->size());
    if (n == 0) return;

    std::vector<int>  label(n, DBSCAN_UNVISITED);
    std::vector<bool> queued(n, false);

    int cluster_id = 0;

    std::vector<int>   nn_indices;
    std::vector<float> nn_dists;
    std::deque<int>    seeds;

    for (int i = 0; i < n; ++i)
    {
        if (label[i] != DBSCAN_UNVISITED) continue;

        nn_indices.clear(); nn_dists.clear();
        tree->radiusSearch(i, eps, nn_indices, nn_dists);

        if (static_cast<int>(nn_indices.size()) < minPts) {
            label[i] = DBSCAN_NOISE;
            continue;
        }

        label[i] = cluster_id;
        int cluster_pts = 1;

        seeds.clear();
        for (int nb : nn_indices) {
            if (nb == i) continue;
            if (!queued[nb]) { queued[nb] = true; seeds.push_back(nb); }
        }

        while (!seeds.empty())
        {
            int q = seeds.front(); seeds.pop_front();

            if (label[q] == DBSCAN_NOISE || label[q] == DBSCAN_UNVISITED) {
                label[q] = cluster_id;
                ++cluster_pts;
            }

            if (label[q] != DBSCAN_NOISE) {
                nn_indices.clear(); nn_dists.clear();
                tree->radiusSearch(q, eps, nn_indices, nn_dists);

                if (static_cast<int>(nn_indices.size()) >= minPts) {
                    for (int nb : nn_indices) {
                        if (!queued[nb] && label[nb] == DBSCAN_UNVISITED) {
                            queued[nb] = true;
                            seeds.push_back(nb);
                        } else if (label[nb] == DBSCAN_NOISE) {
                            label[nb] = cluster_id;
                            ++cluster_pts;
                        }
                    }
                }
            }

            if (cluster_pts > max_cluster_size) break;
        }

        for (int k = 0; k < n; ++k) {
            if (queued[k]) queued[k] = false;
        }

        if (cluster_pts >= min_cluster_size && cluster_pts <= max_cluster_size) {
            pcl::PointIndices ci;
            ci.indices.reserve(cluster_pts);
            for (int k = 0; k < n; ++k) {
                if (label[k] == cluster_id) ci.indices.push_back(k);
            }
            clusters.push_back(std::move(ci));
        }

        ++cluster_id;
    }
}

// ---- Header notes update ----
// EuclideanClusters.cpp (now uses DBSCAN)
// Subscribes : velodyne_points_ground_removed   (PointCloud2, PointXYZ)
// Publishes  : velodyne_points_clustered         (PointCloud2, PointXYZRGB — colour per cluster)

class DBSCANClustering : public rclcpp::Node
{
public:
    DBSCANClustering() : Node("dbscan_clustering")
    {
        this->declare_parameter<double>("cluster_tolerance", 0.50);
        this->declare_parameter<int>("dbscan_min_pts", 5);
        this->declare_parameter<int>("min_cluster_size",  5);
        this->declare_parameter<int>("max_cluster_size",  50000);

        cluster_tolerance_ = this->get_parameter("cluster_tolerance").as_double();
        dbscan_min_pts_    = this->get_parameter("dbscan_min_pts").as_int();
        min_cluster_size_  = this->get_parameter("min_cluster_size").as_int();
        max_cluster_size_  = this->get_parameter("max_cluster_size").as_int();

        // Subscribe to the ground-removed cloud
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "velodyne_points_ground_removed", 10,
            std::bind(&DBSCANClustering::topic_callback, this, std::placeholders::_1));

        // Publish coloured cluster cloud (PointXYZRGB — visible in RViz as colour)
        cluster_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "velodyne_points_clustered", 10);

        RCLCPP_INFO(this->get_logger(),
            "DBSCANClustering started.  eps=%.3f  minPts=%d  min=%d  max=%d",
            cluster_tolerance_, dbscan_min_pts_, min_cluster_size_, max_cluster_size_);
    }

private:
    void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // ---- 1. Convert to PointXYZ for clustering ----
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Received empty cloud — ground removal producing nothing?");
            return;
        }

        RCLCPP_DEBUG(this->get_logger(), "Clustering %zu points …", cloud->size());

        // ---- 2. KdTree + DBSCAN Cluster Extraction ----
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        extractDBSCAN(cloud, tree, cluster_tolerance_, dbscan_min_pts_, min_cluster_size_, max_cluster_size_, cluster_indices);

        if (cluster_indices.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "No clusters found in %zu pts  (tol=%.3f  min=%d). "
                "If this persists check that ground removal is publishing non-empty clouds.",
                cloud->size(), cluster_tolerance_, min_cluster_size_);
            // Publish empty coloured cloud so RViz stops showing stale data
            pcl::PointCloud<pcl::PointXYZRGB> empty;
            sensor_msgs::msg::PointCloud2 out;
            pcl::toROSMsg(empty, out);
            out.header = msg->header;
            cluster_pub_->publish(out);
            return;
        }

        RCLCPP_INFO(this->get_logger(),
            "Found %zu cluster(s) from %zu points.",
            cluster_indices.size(), cloud->size());

        // ---- 3. Colour each cluster and merge into one XYZRGB cloud ----
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloured(new pcl::PointCloud<pcl::PointXYZRGB>);
        coloured->reserve(cloud->size());

        std::size_t colour_idx = 0;
        for (const auto & cluster : cluster_indices)
        {
            const uint8_t * col = PALETTE[colour_idx % N_COLOURS];
            for (const int idx : cluster.indices) {
                pcl::PointXYZRGB pt;
                pt.x = (*cloud)[idx].x;
                pt.y = (*cloud)[idx].y;
                pt.z = (*cloud)[idx].z;
                pt.r = col[0];
                pt.g = col[1];
                pt.b = col[2];
                coloured->push_back(pt);
            }
            ++colour_idx;
        }

        // ---- 4. Publish ----
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*coloured, output);
        output.header = msg->header;   // preserve frame_id and timestamp
        cluster_pub_->publish(output);
    }

    double cluster_tolerance_;
    int    dbscan_min_pts_;
    int    min_cluster_size_;
    int    max_cluster_size_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    cluster_pub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DBSCANClustering>());
    rclcpp::shutdown();
    return 0;
}
