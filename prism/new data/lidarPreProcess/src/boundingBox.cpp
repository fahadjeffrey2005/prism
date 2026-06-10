#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/search/kdtree.h>

#include <vector>
#include <numeric>    
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <deque>        

// Single cluster colour (R, G, B) — change here to restyle all clusters at once
static constexpr uint8_t CLUSTER_R = 0;
static constexpr uint8_t CLUSTER_G = 120;
static constexpr uint8_t CLUSTER_B = 0;   

// ---- Union-Find (path compression + rank) ----
struct UF {
    std::vector<int> parent, rank_;
    explicit UF(int n) : parent(n), rank_(n, 0) { std::iota(parent.begin(), parent.end(), 0); }
    int find(int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
    }
};

// Minimum gap (metres) between two AABBs (0 if they overlap)
static float boxGap(float ax0, float ax1, float ay0, float ay1, float az0, float az1,
                    float bx0, float bx1, float by0, float by1, float bz0, float bz1)
{
    float dx = std::max(0.0f, std::max(ax0, bx0) - std::min(ax1, bx1));
    float dy = std::max(0.0f, std::max(ay0, by0) - std::min(ay1, by1));
    float dz = std::max(0.0f, std::max(az0, bz0) - std::min(az1, bz1));
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// ---- Range-Adaptive DBSCAN ----
//
// LiDAR density drops ~1/r^2 with range. A human at 20 m has far fewer
// points than at 5 m.  Three thresholds scale linearly with distance:
//
//   eps_i      = eps_near * (1 + eps_range_factor * r_i)
//   minPts_i   = lerp(min_pts_near, min_pts_far,  clamp01(r_i / far_range))
//   min_size_i = lerp(min_size_near, min_size_far, clamp01(r_i / far_cluster_range))
//
static constexpr int DBSCAN_UNVISITED = -1;
static constexpr int DBSCAN_NOISE     = -2;

struct DBSCANParams {
    float eps_near;          // search radius (m) at r == 0
    float eps_range_factor;  // fractional growth per metre (0.05 = 5%/m)
    int   min_pts_near;      // minPts at close range
    int   min_pts_far;       // minPts at >= far_range
    float far_range;         // metres where min_pts_far applies
    int   min_size_near;     // min cluster size near
    int   min_size_far;      // min cluster size far
    float far_cluster_range; // metres where min_size_far applies
    int   max_cluster_size;
};

void extractDBSCAN(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
    const pcl::search::KdTree<pcl::PointXYZ>::Ptr & tree,
    const DBSCANParams & p,
    std::vector<pcl::PointIndices> & clusters)
{
    const int n = static_cast<int>(cloud->size());
    if (n == 0) return;

    // Pre-compute per-point range-adaptive eps and minPts
    std::vector<float> eps_pt(n), range_pt(n);
    std::vector<int>   minPts_pt(n);
    for (int i = 0; i < n; ++i) {
        const auto & pt = (*cloud)[i];
        float r = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
        range_pt[i]  = r;
        eps_pt[i]    = p.eps_near * (1.0f + p.eps_range_factor * r);
        float t      = std::min(1.0f, r / std::max(p.far_range, 0.1f));
        minPts_pt[i] = std::max(1, static_cast<int>(std::round(
                           p.min_pts_near * (1.0f - t) + p.min_pts_far * t)));
    }

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
        tree->radiusSearch(i, eps_pt[i], nn_indices, nn_dists);

        if (static_cast<int>(nn_indices.size()) < minPts_pt[i]) {
            label[i] = DBSCAN_NOISE;
            continue;
        }

        // Adaptive min cluster size for this cluster (based on seed point range)
        float tc = std::min(1.0f, range_pt[i] / std::max(p.far_cluster_range, 0.1f));
        int min_size_i = std::max(1, static_cast<int>(std::round(
                             p.min_size_near * (1.0f - tc) + p.min_size_far * tc)));

        label[i] = cluster_id;
        int cluster_pts = 1;

        // seed the expansion queue (exclude i itself which is already labelled)
        seeds.clear();
        for (int nb : nn_indices) {
            if (nb == i) continue;
            if (!queued[nb]) {
                queued[nb] = true;
                seeds.push_back(nb);
            }
        }

        while (!seeds.empty())
        {
            int q = seeds.front(); seeds.pop_front();

            // absorb noise/unvisited point into the current cluster
            if (label[q] == DBSCAN_NOISE || label[q] == DBSCAN_UNVISITED) {
                label[q] = cluster_id;
                ++cluster_pts;
            }
            // if q was already claimed by this cluster (border point reached twice)
            // we still need to check if it's a core and expand, but only if unvisited

            if (label[q] != DBSCAN_NOISE)
            {
                nn_indices.clear(); nn_dists.clear();
                tree->radiusSearch(q, eps_pt[q], nn_indices, nn_dists);

                if (static_cast<int>(nn_indices.size()) >= minPts_pt[q]) {
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

            // Early exit if cluster already exceeds max size
            if (cluster_pts > p.max_cluster_size) break;
        }

        // Clear queued flags for all points we seeded (even if we broke early)
        // We walk the cloud labels rather than keeping a separate "dirtied" list,
        // because resetting just the affected indices is cheaper than a full clear.
        for (int k = 0; k < n; ++k) {
            if (queued[k]) queued[k] = false;
        }

        // Collect cluster if within size bounds
        if (cluster_pts >= min_size_i && cluster_pts <= p.max_cluster_size) {
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

// ---- Node ----
class BoundingBoxNode : public rclcpp::Node
{
public:
    BoundingBoxNode() : Node("bounding_box_node"), frame_counter_(0)
    {
        // Clustering — range-adaptive DBSCAN
        this->declare_parameter<double>("cluster_tolerance",    0.20); // eps at range=0
        this->declare_parameter<double>("eps_range_factor",     0.05); // eps grows 5% per metre
        this->declare_parameter<int>   ("dbscan_min_pts",          5); // minPts near
        this->declare_parameter<int>   ("min_pts_far",              2); // minPts at far_range
        this->declare_parameter<double>("far_range",            20.0); // m
        this->declare_parameter<int>   ("min_cluster_size",       30); // near
        this->declare_parameter<int>   ("min_cluster_size_far",    5); // far
        this->declare_parameter<double>("far_cluster_range",    20.0); // m
        this->declare_parameter<int>   ("max_cluster_size",    50000);
        // density threshold also scales down with range
        this->declare_parameter<double>("density_range_factor", 0.05); // divides threshold per metre

        // Z passthrough
        this->declare_parameter<double>("z_min", -10.0);
        this->declare_parameter<double>("z_max",  50.0);

        // Per-cluster filters
        this->declare_parameter<double>("max_box_dimension",  15.0);  // m — any axis
        this->declare_parameter<double>("min_box_volume",      0.01); // m³
        this->declare_parameter<double>("max_ground_clearance", 0.5); // m

        this->declare_parameter<double>("min_point_density",   5.0);  // pts/m³
        this->declare_parameter<double>("merge_distance",      1.5);  // m

        // Visualisation
        this->declare_parameter<double>("marker_alpha",  0.60);

        // Performance: process 1-in-N frames
        this->declare_parameter<int>("skip_frames", 2);

        cluster_tolerance_    = this->get_parameter("cluster_tolerance").as_double();
        eps_range_factor_     = this->get_parameter("eps_range_factor").as_double();
        dbscan_min_pts_       = this->get_parameter("dbscan_min_pts").as_int();
        min_pts_far_          = this->get_parameter("min_pts_far").as_int();
        far_range_            = this->get_parameter("far_range").as_double();
        min_cluster_size_     = this->get_parameter("min_cluster_size").as_int();
        min_cluster_size_far_ = this->get_parameter("min_cluster_size_far").as_int();
        far_cluster_range_    = this->get_parameter("far_cluster_range").as_double();
        max_cluster_size_     = this->get_parameter("max_cluster_size").as_int();
        density_range_factor_ = this->get_parameter("density_range_factor").as_double();
        z_min_                = this->get_parameter("z_min").as_double();
        z_max_                = this->get_parameter("z_max").as_double();
        max_box_dimension_    = this->get_parameter("max_box_dimension").as_double();
        min_box_volume_       = this->get_parameter("min_box_volume").as_double();
        max_ground_clearance_ = this->get_parameter("max_ground_clearance").as_double();
        min_point_density_    = this->get_parameter("min_point_density").as_double();
        merge_distance_       = this->get_parameter("merge_distance").as_double();
        marker_alpha_         = this->get_parameter("marker_alpha").as_double();
        skip_frames_          = std::max(1, static_cast<int>(
                                    this->get_parameter("skip_frames").as_int()));

        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "velodyne_points_ground_removed", qos,
            std::bind(&BoundingBoxNode::topic_callback, this, std::placeholders::_1));

        cluster_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "velodyne_points_clustered", rclcpp::QoS(1));
        bbox_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "cluster_bounding_boxes", rclcpp::QoS(1));

        RCLCPP_INFO(this->get_logger(),
            "BoundingBoxNode ready (range-adaptive DBSCAN)\n"
            "  eps=%.2f factor=%.3f | min_pts near=%d far=%d at %.0fm\n"
            "  min_cluster near=%d far=%d at %.0fm\n"
            "  z=[%.1f,%.1f]  gnd_clear=%.2f\n"
            "  min_density=%.1f pts/m3 (factor %.3f)  merge=%.2fm\n"
            "  alpha=%.2f  skip=%d",
            cluster_tolerance_, eps_range_factor_,
            dbscan_min_pts_, min_pts_far_, far_range_,
            min_cluster_size_, min_cluster_size_far_, far_cluster_range_,
            z_min_, z_max_, max_ground_clearance_,
            min_point_density_, density_range_factor_, merge_distance_,
            marker_alpha_, skip_frames_);
    }

private:
    // ------------------------------------------------------------------
    void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if ((++frame_counter_ % skip_frames_) != 0) return;

        // ---- 1. Convert ----
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);
        if (cloud->empty()) return;

        // ---- 2. Z passthrough ----
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_z;
        if (z_min_ > -9.9 || z_max_ < 49.9) {
            cloud_z.reset(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PassThrough<pcl::PointXYZ> pass;
            pass.setInputCloud(cloud);
            pass.setFilterFieldName("z");
            pass.setFilterLimits(static_cast<float>(z_min_), static_cast<float>(z_max_));
            pass.filter(*cloud_z);
        } else {
            cloud_z = cloud;
        }
        if (cloud_z->empty()) {
            bbox_pub_->publish(visualization_msgs::msg::MarkerArray{});
            return;
        }

        // ---- 3. DBSCAN clustering ----
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud_z);
        std::vector<pcl::PointIndices> raw_clusters;
        
        DBSCANParams dp;
        dp.eps_near          = static_cast<float>(cluster_tolerance_);
        dp.eps_range_factor  = static_cast<float>(eps_range_factor_);
        dp.min_pts_near      = dbscan_min_pts_;
        dp.min_pts_far       = min_pts_far_;
        dp.far_range         = static_cast<float>(far_range_);
        dp.min_size_near     = min_cluster_size_;
        dp.min_size_far      = min_cluster_size_far_;
        dp.far_cluster_range = static_cast<float>(far_cluster_range_);
        dp.max_cluster_size  = max_cluster_size_;
        extractDBSCAN(cloud_z, tree, dp, raw_clusters);

        if (raw_clusters.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                "No clusters in %zu pts (tol=%.2f min=%d)",
                cloud_z->size(), cluster_tolerance_, min_cluster_size_);
            bbox_pub_->publish(visualization_msgs::msg::MarkerArray{});
            return;
        }

        // ---- 4. Per-cluster filters → collect surviving cluster AABBs ----
        struct ClusterBox {
            std::vector<int> indices;
            float x0, x1, y0, y1, z0, z1;
        };
        std::vector<ClusterBox> survivors;
        survivors.reserve(raw_clusters.size());

        for (const auto & ci : raw_clusters)
        {
            float xmn= 1e9f, ymn= 1e9f, zmn= 1e9f;
            float xmx=-1e9f, ymx=-1e9f, zmx=-1e9f;
            for (int idx : ci.indices) {
                const auto & p = (*cloud_z)[idx];
                if (p.x < xmn) { xmn = p.x; } if (p.x > xmx) { xmx = p.x; }
                if (p.y < ymn) { ymn = p.y; } if (p.y > ymx) { ymx = p.y; }
                if (p.z < zmn) { zmn = p.z; } if (p.z > zmx) { zmx = p.z; }
            }
            double sx = xmx-xmn, sy = ymx-ymn, sz = zmx-zmn;
            double vol = std::max(sx*sy*sz, 1e-6);  // avoid /0

            // a) Ground-clearance filter
            if (static_cast<double>(zmn) > max_ground_clearance_) continue;
            // b) Size sanity
            if (sx > max_box_dimension_ || sy > max_box_dimension_ ||
                sz > max_box_dimension_) continue;
            if (vol < min_box_volume_) continue;
            // c) Density filter: threshold scales with range so far clusters aren't dropped
            double cx_cl = 0.5*(xmn+xmx), cy_cl = 0.5*(ymn+ymx), cz_cl = 0.5*(zmn+zmx);
            double cluster_range = std::sqrt(cx_cl*cx_cl + cy_cl*cy_cl + cz_cl*cz_cl);
            double eff_thresh = min_point_density_ / (1.0 + density_range_factor_ * cluster_range);
            double density = static_cast<double>(ci.indices.size()) / vol;
            if (density < eff_thresh) {
                RCLCPP_DEBUG(this->get_logger(),
                    "Noise cluster: %.1f < %.1f pts/m3 at range %.1fm",
                    density, eff_thresh, cluster_range);
                continue;
            }

            ClusterBox cb;
            cb.indices = ci.indices;
            cb.x0 = xmn; cb.x1 = xmx;
            cb.y0 = ymn; cb.y1 = ymx;
            cb.z0 = zmn; cb.z1 = zmx;
            survivors.push_back(std::move(cb));
        }

        if (survivors.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                "All clusters filtered out (raw=%zu). "
                "Loosen min_point_density / min_cluster_size / max_ground_clearance.",
                raw_clusters.size());
            bbox_pub_->publish(visualization_msgs::msg::MarkerArray{});
            return;
        }

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "raw=%zu → filtered=%zu clusters",
            raw_clusters.size(), survivors.size());

        const int N = static_cast<int>(survivors.size());
        UF uf(N);
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                float gap = boxGap(
                    survivors[i].x0, survivors[i].x1,
                    survivors[i].y0, survivors[i].y1,
                    survivors[i].z0, survivors[i].z1,
                    survivors[j].x0, survivors[j].x1,
                    survivors[j].y0, survivors[j].y1,
                    survivors[j].z0, survivors[j].z1);
                if (gap <= static_cast<float>(merge_distance_)) {
                    uf.unite(i, j);
                }
            }
        }

        // Group survivors by root
        std::unordered_map<int, std::vector<int>> groups;
        for (int i = 0; i < N; ++i) groups[uf.find(i)].push_back(i);

        // ---- 6. Build merged AABBs + coloured cloud + markers ----
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloured(new pcl::PointCloud<pcl::PointXYZRGB>);
        coloured->reserve(cloud_z->size());

        visualization_msgs::msg::MarkerArray marker_array;
        const rclcpp::Duration lifetime = rclcpp::Duration::from_seconds(
            0.6 * static_cast<double>(skip_frames_));

        int marker_id   = 0;
        int drawn_count = 0;

        auto mkpt = [](double x, double y, double z) {
            geometry_msgs::msg::Point p; p.x = x; p.y = y; p.z = z; return p;
        };

        for (auto & [root, members] : groups)
        {
            // Merged AABB over all member clusters
            float X0= 1e9f, Y0= 1e9f, Z0= 1e9f;
            float X1=-1e9f, Y1=-1e9f, Z1=-1e9f;
            for (int m : members) {
                X0 = std::min(X0, survivors[m].x0); X1 = std::max(X1, survivors[m].x1);
                Y0 = std::min(Y0, survivors[m].y0); Y1 = std::max(Y1, survivors[m].y1);
                Z0 = std::min(Z0, survivors[m].z0); Z1 = std::max(Z1, survivors[m].z1);
            }

            double sx = std::max((double)(X1-X0), 0.1);
            double sy = std::max((double)(Y1-Y0), 0.1);
            double sz = std::max((double)(Z1-Z0), 0.1);
            double cx = 0.5*(X0+X1), cy = 0.5*(Y0+Y1), cz = 0.5*(Z0+Z1);

            constexpr float r = CLUSTER_R / 255.0f;
            constexpr float g = CLUSTER_G / 255.0f;
            constexpr float b = CLUSTER_B / 255.0f;

            // Colour all points belonging to this merged group
            for (int m : members) {
                for (int idx : survivors[m].indices) {
                    const auto & p = (*cloud_z)[idx];
                    pcl::PointXYZRGB cp;
                    cp.x=p.x; cp.y=p.y; cp.z=p.z;
                    cp.r=CLUSTER_R; cp.g=CLUSTER_G; cp.b=CLUSTER_B;
                    coloured->push_back(cp);
                }
            }

            // ---- Filled CUBE ----
            {
                visualization_msgs::msg::Marker box;
                box.header = msg->header;
                box.ns     = "bbox_fill";
                box.id     = marker_id;
                box.type   = visualization_msgs::msg::Marker::CUBE;
                box.action = visualization_msgs::msg::Marker::ADD;
                box.pose.position.x = cx;
                box.pose.position.y = cy;
                box.pose.position.z = cz;
                box.pose.orientation.w = 1.0;
                box.scale.x = sx; box.scale.y = sy; box.scale.z = sz;
                box.color.r = r; box.color.g = g; box.color.b = b;
                box.color.a = static_cast<float>(marker_alpha_);
                box.lifetime = lifetime;
                marker_array.markers.push_back(box);
            }

            // ---- Wireframe LINE_LIST ----
            {
                visualization_msgs::msg::Marker wire;
                wire.header = msg->header;
                wire.ns     = "bbox_wire";
                wire.id     = marker_id + 1;
                wire.type   = visualization_msgs::msg::Marker::LINE_LIST;
                wire.action = visualization_msgs::msg::Marker::ADD;
                wire.pose.orientation.w = 1.0;
                wire.scale.x = 0.04f;
                wire.color.r = r; wire.color.g = g; wire.color.b = b;
                wire.color.a = 1.0f;
                wire.lifetime = lifetime;

                double x0=X0, x1=X1, y0=Y0, y1=Y1, z0=Z0, z1=Z1;
                wire.points.push_back(mkpt(x0,y0,z0)); wire.points.push_back(mkpt(x1,y0,z0));
                wire.points.push_back(mkpt(x1,y0,z0)); wire.points.push_back(mkpt(x1,y1,z0));
                wire.points.push_back(mkpt(x1,y1,z0)); wire.points.push_back(mkpt(x0,y1,z0));
                wire.points.push_back(mkpt(x0,y1,z0)); wire.points.push_back(mkpt(x0,y0,z0));
                wire.points.push_back(mkpt(x0,y0,z1)); wire.points.push_back(mkpt(x1,y0,z1));
                wire.points.push_back(mkpt(x1,y0,z1)); wire.points.push_back(mkpt(x1,y1,z1));
                wire.points.push_back(mkpt(x1,y1,z1)); wire.points.push_back(mkpt(x0,y1,z1));
                wire.points.push_back(mkpt(x0,y1,z1)); wire.points.push_back(mkpt(x0,y0,z1));
                wire.points.push_back(mkpt(x0,y0,z0)); wire.points.push_back(mkpt(x0,y0,z1));
                wire.points.push_back(mkpt(x1,y0,z0)); wire.points.push_back(mkpt(x1,y0,z1));
                wire.points.push_back(mkpt(x1,y1,z0)); wire.points.push_back(mkpt(x1,y1,z1));
                wire.points.push_back(mkpt(x0,y1,z0)); wire.points.push_back(mkpt(x0,y1,z1));

                marker_array.markers.push_back(wire);
            }

            marker_id += 2;
            (void)drawn_count;  // unused — single colour mode
        }

        // ---- 7. Publish ----
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*coloured, cloud_msg);
        cloud_msg.header = msg->header;
        cluster_pub_->publish(cloud_msg);
        bbox_pub_->publish(marker_array);
    }

    // Parameters
    double cluster_tolerance_;
    double eps_range_factor_;
    int    dbscan_min_pts_, min_pts_far_;
    double far_range_;
    int    min_cluster_size_, min_cluster_size_far_, max_cluster_size_;
    double far_cluster_range_;
    double z_min_, z_max_;
    double max_box_dimension_, min_box_volume_;
    double max_ground_clearance_;
    double min_point_density_;
    double density_range_factor_;
    double merge_distance_;
    double marker_alpha_;
    int    skip_frames_;

    int frame_counter_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    cluster_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr bbox_pub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BoundingBoxNode>());
    rclcpp::shutdown();
    return 0;
}
