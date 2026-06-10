#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cmath>
#include <unordered_map>
#include <vector>
#include <array>
#include <limits>

// Encode (ix, iy) as a single 64-bit key
static inline uint64_t cellKey(int ix, int iy)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(ix)) << 32) |
            static_cast<uint32_t>(iy);
}
static inline int keyIX(uint64_t k) { return static_cast<int>(static_cast<uint32_t>(k >> 32)); }
static inline int keyIY(uint64_t k) { return static_cast<int>(static_cast<uint32_t>(k & 0xFFFFFFFF)); }

class LidarGroundRemoval : public rclcpp::Node
{
public:
    LidarGroundRemoval() : Node("lidar_ground_removal")
    {
        this->declare_parameter<double>("axis_nx",        0.0);
        this->declare_parameter<double>("axis_ny",        0.0);
        this->declare_parameter<double>("axis_nz",        1.0);
        this->declare_parameter<double>("grid_res",       0.5);
        this->declare_parameter<double>("height_margin",  0.25);
        this->declare_parameter<double>("max_range",      40.0);
        this->declare_parameter<int>   ("dilation_steps", 3);
        this->declare_parameter<int>   ("min_pts_in_cell",1);

        double nx = this->get_parameter("axis_nx").as_double();
        double ny = this->get_parameter("axis_ny").as_double();
        double nz = this->get_parameter("axis_nz").as_double();
        double len = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (len < 1e-6) { nx=0.0; ny=0.0; nz=1.0; len=1.0; }
        up_[0]=nx/len; up_[1]=ny/len; up_[2]=nz/len;

        // Build two tangent vectors orthonormal to up_
        double ref[3] = {1.0, 0.0, 0.0};
        if (std::abs(up_[0]) > 0.9) { ref[0]=0.0; ref[1]=1.0; }
        // t1 = normalise(cross(up, ref))
        t1_[0] = up_[1]*ref[2] - up_[2]*ref[1];
        t1_[1] = up_[2]*ref[0] - up_[0]*ref[2];
        t1_[2] = up_[0]*ref[1] - up_[1]*ref[0];
        double tlen = std::sqrt(t1_[0]*t1_[0]+t1_[1]*t1_[1]+t1_[2]*t1_[2]);
        t1_[0]/=tlen; t1_[1]/=tlen; t1_[2]/=tlen;
        // t2 = cross(up, t1)  — already unit length
        t2_[0] = up_[1]*t1_[2] - up_[2]*t1_[1];
        t2_[1] = up_[2]*t1_[0] - up_[0]*t1_[2];
        t2_[2] = up_[0]*t1_[1] - up_[1]*t1_[0];

        grid_res_        = this->get_parameter("grid_res").as_double();
        height_margin_   = this->get_parameter("height_margin").as_double();
        max_range_sq_    = std::pow(this->get_parameter("max_range").as_double(), 2.0);
        dilation_steps_  = this->get_parameter("dilation_steps").as_int();
        min_pts_in_cell_ = this->get_parameter("min_pts_in_cell").as_int();

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "velodyne_points_sor_filtered", 10,
            std::bind(&LidarGroundRemoval::callback, this, std::placeholders::_1));
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "velodyne_points_ground_removed", 10);

        RCLCPP_INFO(this->get_logger(),
            "LidarGroundRemoval (grid-adaptive) ready.\n"
            "  up=[%.3f,%.3f,%.3f]  grid_res=%.2f  height_margin=%.2f\n"
            "  max_range=%.1f  dilation=%d  min_pts=%d",
            up_[0], up_[1], up_[2], grid_res_, height_margin_,
            std::sqrt(max_range_sq_), dilation_steps_, min_pts_in_cell_);
    }

private:
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);
        if (cloud->empty()) { publisher_->publish(*msg); return; }

        const std::size_t N = cloud->size();
        const double inv_res = 1.0 / grid_res_;

        // Per-point: height h and 2D grid index
        std::vector<float> h_vec(N);
        std::vector<int>   ix_vec(N), iy_vec(N);
        std::vector<bool>  in_range(N, true);

        // --- Step 1: project every point ---
        // Accumulate per-cell: sum of h and count (for min computation)
        // Use a map: key → (min_h, count)
        struct CellInfo { float min_h; int cnt; };
        std::unordered_map<uint64_t, CellInfo> grid;
        grid.reserve(N / 4);

        for (std::size_t i = 0; i < N; ++i)
        {
            const float px = cloud->points[i].x;
            const float py = cloud->points[i].y;
            const float pz = cloud->points[i].z;

            // Range check
            double r2 = px*px + py*py + pz*pz;
            if (r2 > max_range_sq_) { in_range[i] = false; continue; }

            // Project onto local frame
            float h  = static_cast<float>(up_[0]*px + up_[1]*py + up_[2]*pz);
            float gx = static_cast<float>(t1_[0]*px + t1_[1]*py + t1_[2]*pz);
            float gy = static_cast<float>(t2_[0]*px + t2_[1]*py + t2_[2]*pz);

            h_vec[i]  = h;
            int ix = static_cast<int>(std::floor(gx * inv_res));
            int iy = static_cast<int>(std::floor(gy * inv_res));
            ix_vec[i] = ix; iy_vec[i] = iy;

            uint64_t key = cellKey(ix, iy);
            auto it = grid.find(key);
            if (it == grid.end()) {
                grid[key] = {h, 1};
            } else {
                if (h < it->second.min_h) it->second.min_h = h;
                it->second.cnt++;
            }
        }

        // --- Step 2: remove cells with too few points ---
        if (min_pts_in_cell_ > 1) {
            for (auto it = grid.begin(); it != grid.end(); ) {
                if (it->second.cnt < min_pts_in_cell_) it = grid.erase(it);
                else ++it;
            }
        }

        // --- Step 3: dilation — fill empty neighbour cells ---
        static const int DX[4] = {-1, 1, 0, 0};
        static const int DY[4] = { 0, 0,-1, 1};

        for (int step = 0; step < dilation_steps_; ++step)
        {
            std::vector<std::pair<uint64_t, float>> to_add;
            to_add.reserve(grid.size() * 2);

            for (const auto & [key, ci] : grid) {
                int ix = keyIX(key), iy = keyIY(key);
                for (int d = 0; d < 4; ++d) {
                    uint64_t nk = cellKey(ix + DX[d], iy + DY[d]);
                    if (grid.find(nk) == grid.end()) {
                        to_add.push_back({nk, ci.min_h});
                    }
                }
            }
            for (auto & [k, v] : to_add) {
                // Only insert if still absent (first-come wins = nearest-source wins)
                grid.emplace(k, CellInfo{v, 0});
            }
        }

        // --- Step 4: classify and publish non-ground points ---
        pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground(new pcl::PointCloud<pcl::PointXYZ>);
        non_ground->reserve(N);

        std::size_t removed = 0;
        for (std::size_t i = 0; i < N; ++i)
        {
            if (!in_range[i]) {
                // Out-of-range points: keep them (obstacle detection at long range)
                non_ground->push_back(cloud->points[i]);
                continue;
            }

            uint64_t key = cellKey(ix_vec[i], iy_vec[i]);
            auto it = grid.find(key);
            if (it == grid.end()) {
                // No ground reference in this cell — keep the point
                non_ground->push_back(cloud->points[i]);
                continue;
            }

            float local_ground = it->second.min_h;
            if (h_vec[i] > local_ground + static_cast<float>(height_margin_)) {
                non_ground->push_back(cloud->points[i]);
            } else {
                ++removed;
            }
        }

        sensor_msgs::msg::PointCloud2 out;
        pcl::toROSMsg(*non_ground, out);
        out.header = msg->header;
        publisher_->publish(out);

        RCLCPP_DEBUG(this->get_logger(),
            "Grid ground filter: %zu → %zu pts (%zu removed, %zu grid cells)",
            N, non_ground->size(), removed, grid.size());
    }

    // Up axis and two tangent axes (orthonormal frame)
    double up_[3], t1_[3], t2_[3];
    double grid_res_, height_margin_, max_range_sq_;
    int    dilation_steps_, min_pts_in_cell_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarGroundRemoval>());
    rclcpp::shutdown();
    return 0;
}
