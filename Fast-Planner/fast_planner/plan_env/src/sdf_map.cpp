/**
* This file is part of Fast-Planner.
*
* Copyright 2019 Boyu Zhou, Aerial Robotics Group, Hong Kong University of Science and Technology, <uav.ust.hk>
* Developed by Boyu Zhou <bzhouai at connect dot ust dot hk>, <uv dot boyuzhou at gmail dot com>
* for more information see <https://github.com/HKUST-Aerial-Robotics/Fast-Planner>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* Fast-Planner is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Fast-Planner is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with Fast-Planner. If not, see <http://www.gnu.org/licenses/>.
*/



#include "plan_env/sdf_map.h"

// #define current_img_ md_.depth_image_[image_cnt_ & 1]
// #define last_img_ md_.depth_image_[!(image_cnt_ & 1)]

void SDFMap::initMap(ros::NodeHandle& nh,ros::NodeHandle& map_nh) {
  node_ = nh;
  map_node_ = map_nh;
  
  /* get parameter */
  double x_size, y_size, z_size;
  
  node_.param("sdf_map/resolution", mp_.resolution_, -1.0);
  node_.param("sdf_map/map_size_x", x_size, -1.0);
  node_.param("sdf_map/map_size_y", y_size, -1.0);
  node_.param("sdf_map/map_size_z", z_size, -1.0);
  node_.param("sdf_map/local_update_range_x", mp_.local_update_range_(0), -1.0);
  node_.param("sdf_map/local_update_range_y", mp_.local_update_range_(1), -1.0);
  node_.param("sdf_map/local_update_range_z", mp_.local_update_range_(2), -1.0);
  node_.param("sdf_map/obstacles_inflation", mp_.obstacles_inflation_, -1.0);

  node_.param("sdf_map/fx", mp_.fx_, -1.0);
  node_.param("sdf_map/fy", mp_.fy_, -1.0);
  node_.param("sdf_map/cx", mp_.cx_, -1.0);
  node_.param("sdf_map/cy", mp_.cy_, -1.0);

  node_.param("sdf_map/use_depth_filter", mp_.use_depth_filter_, true);
  node_.param("sdf_map/depth_filter_tolerance", mp_.depth_filter_tolerance_, -1.0);
  node_.param("sdf_map/depth_filter_maxdist", mp_.depth_filter_maxdist_, -1.0);
  node_.param("sdf_map/depth_filter_mindist", mp_.depth_filter_mindist_, -1.0);
  node_.param("sdf_map/depth_filter_margin", mp_.depth_filter_margin_, -1);
  node_.param("sdf_map/k_depth_scaling_factor", mp_.k_depth_scaling_factor_, -1.0);
  node_.param("sdf_map/skip_pixel", mp_.skip_pixel_, -1);

  node_.param("sdf_map/p_hit", mp_.p_hit_, 0.70);
  node_.param("sdf_map/p_miss", mp_.p_miss_, 0.35);
  node_.param("sdf_map/p_min", mp_.p_min_, 0.12);
  node_.param("sdf_map/p_max", mp_.p_max_, 0.97);
  node_.param("sdf_map/p_occ", mp_.p_occ_, 0.80);
  node_.param("sdf_map/min_ray_length", mp_.min_ray_length_, -0.1);
  node_.param("sdf_map/max_ray_length", mp_.max_ray_length_, -0.1);

  node_.param("sdf_map/esdf_slice_height", mp_.esdf_slice_height_, -0.1);
  node_.param("sdf_map/visualization_truncate_height", mp_.visualization_truncate_height_, -0.1);
  node_.param("sdf_map/virtual_ceil_height", mp_.virtual_ceil_height_, -0.1);

  node_.param("sdf_map/show_occ_time", mp_.show_occ_time_, false);
  node_.param("sdf_map/show_esdf_time", mp_.show_esdf_time_, false);
  node_.param("sdf_map/pose_type", mp_.pose_type_, 2);

  node_.param("sdf_map/frame_id", mp_.frame_id_, string("world"));
  node_.param("sdf_map/robot_frame_id", robot_frame_, string("camera_link"));
  node_.param("sdf_map/world_frame_id", world_frame_, string("world"));  
  node_.param("sdf_map/local_bound_inflate", mp_.local_bound_inflate_, 1.0);
  node_.param("sdf_map/local_map_margin", mp_.local_map_margin_, 1);
  node_.param("sdf_map/ground_height", mp_.ground_height_, 1.0);
  node_.param("sdf_map/sensor_fov", sensor_FOV, 1.456);
  

  mp_.local_bound_inflate_ = max(mp_.resolution_, mp_.local_bound_inflate_);
  mp_.resolution_inv_ = 1 / mp_.resolution_;
  mp_.map_origin_ = Eigen::Vector3d(-x_size / 2.0, -y_size / 2.0, mp_.ground_height_);  
  mp_.map_size_ = Eigen::Vector3d(x_size, y_size, z_size);

  mp_.prob_hit_log_ = logit(mp_.p_hit_);
  mp_.prob_miss_log_ = logit(mp_.p_miss_);
  mp_.clamp_min_log_ = logit(mp_.p_min_);
  mp_.clamp_max_log_ = logit(mp_.p_max_);
  mp_.min_occupancy_log_ = logit(mp_.p_occ_);
  mp_.unknown_flag_ = 0.01;

  cout << "hit: " << mp_.prob_hit_log_ << endl;
  cout << "miss: " << mp_.prob_miss_log_ << endl;
  cout << "min log: " << mp_.clamp_min_log_ << endl;
  cout << "max: " << mp_.clamp_max_log_ << endl;
  cout << "thresh log: " << mp_.min_occupancy_log_ << endl;

  for (int i = 0; i < 3; ++i) mp_.map_voxel_num_(i) = ceil(mp_.map_size_(i) / mp_.resolution_);

  mp_.map_min_boundary_ = mp_.map_origin_;
  mp_.map_max_boundary_ = mp_.map_origin_ + mp_.map_size_;

  mp_.map_min_idx_ = Eigen::Vector3i::Zero();
  mp_.map_max_idx_ = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();

  // initialize data buffers

  int buffer_size = mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2);

  md_.occupancy_buffer_ = vector<double>(buffer_size, mp_.clamp_min_log_ - mp_.unknown_flag_);
  md_.occupancy_buffer_neg = vector<char>(buffer_size, 0);
  md_.occupancy_buffer_inflate_ = vector<char>(buffer_size, 0);

  md_.distance_buffer_ = vector<double>(buffer_size, 10000);
  md_.distance_buffer_neg_ = vector<double>(buffer_size, 10000);
  md_.distance_buffer_all_ = vector<double>(buffer_size, 10000);

  md_.count_hit_and_miss_ = vector<short>(buffer_size, 0);
  md_.count_hit_ = vector<short>(buffer_size, 0);
  md_.flag_rayend_ = vector<char>(buffer_size, -1);
  md_.flag_traverse_ = vector<char>(buffer_size, -1);

  md_.tmp_buffer1_ = vector<double>(buffer_size, 0);
  md_.tmp_buffer2_ = vector<double>(buffer_size, 0);
  md_.raycast_num_ = 0;

  md_.proj_points_.resize(640 * 480 / mp_.skip_pixel_ / mp_.skip_pixel_);
  md_.proj_points_cnt = 0;

  /* init callback */
  global_direction = 0.0;
  odom_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(map_node_, "/mavros/local_position/odom", 100));
    lidar_sub_.reset(new message_filters::Subscriber<sensor_msgs::LaserScan>(map_node_, "/laser/scan", 100));
    point_cloud_sub_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(map_node_, "/camera/depth/color/points", 100));

    oned_lidar_cloud_sub_.reset(new message_filters::Subscriber<sensor_msgs::Range>(map_node_, "/mavros/distance_sensor/lidarlite_pub", 100));

    

    sync_pose_laser_points_.reset(new message_filters::Synchronizer<SyncPolicyPoseLaserPoints>(
        SyncPolicyPoseLaserPoints(100), *odom_sub_, *lidar_sub_, *point_cloud_sub_, *oned_lidar_cloud_sub_));
    sync_pose_laser_points_->registerCallback(boost::bind(&SDFMap::odomLaserCloudCallback, this, _1, _2, _3, _4));


  
  // occ_timer_ = node_.createTimer(ros::Duration(0.05), &SDFMap::updateOccupancyCallback, this);
  esdf_timer_ = node_.createTimer(ros::Duration(0.05), &SDFMap::updateESDFCallback, this);
  vis_timer_ = node_.createTimer(ros::Duration(0.05), &SDFMap::visCallback, this);

  map_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy", 10);
  map_inf_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_inflate", 10);
  esdf_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/esdf", 10);
  update_range_pub_ = node_.advertise<visualization_msgs::Marker>("/sdf_map/update_range", 10);
  goal_direction_pub_ = node_.advertise<std_msgs::Float32>("goal_direction",10);

  unknown_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/unknown", 10);
  depth_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/depth_cloud", 10);

  md_.occ_need_update_ = false;
  md_.local_updated_ = false;
  md_.esdf_need_update_ = false;
  md_.has_first_depth_ = false;
  md_.has_odom_ = false;
  md_.has_cloud_ = false;
  md_.image_cnt_ = 0;

  md_.esdf_time_ = 0.0;
  md_.fuse_time_ = 0.0;
  md_.update_num_ = 0;
  md_.max_esdf_time_ = 0.0;
  md_.max_fuse_time_ = 0.0;

  rand_noise_ = uniform_real_distribution<double>(-0.2, 0.2);
  rand_noise2_ = normal_distribution<double>(0, 0.2);
  random_device rd;
  eng_ = default_random_engine(rd());
}


void SDFMap::resetBuffer() {
  Eigen::Vector3d min_pos = mp_.map_min_boundary_;
  Eigen::Vector3d max_pos = mp_.map_max_boundary_;

  resetBuffer(min_pos, max_pos);

  md_.local_bound_min_ = Eigen::Vector3i::Zero();
  md_.local_bound_max_ = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();
}


void SDFMap::resetBuffer(Eigen::Vector3d min_pos, Eigen::Vector3d max_pos) {


  const int vec_margin = 5;
  // Eigen::Vector3i min_vec_margin = min_vec - Eigen::Vector3i(vec_margin,
  // vec_margin, vec_margin); Eigen::Vector3i max_vec_margin = max_vec +
  // Eigen::Vector3i(vec_margin, vec_margin, vec_margin);

  Eigen::Vector3i min_cut = md_.local_bound_min_ -
      Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
  Eigen::Vector3i max_cut = md_.local_bound_max_ +
      Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
  boundIndex(min_cut);
  boundIndex(max_cut);

  Eigen::Vector3i min_cut_m = min_cut - Eigen::Vector3i(vec_margin, vec_margin, vec_margin);
  Eigen::Vector3i max_cut_m = max_cut + Eigen::Vector3i(vec_margin, vec_margin, vec_margin);
  boundIndex(min_cut_m);
  boundIndex(max_cut_m);

 

  // clear data outside the local range
  
  for (int x = min_cut_m(0); x <= max_cut_m(0); ++x)
    for (int y = min_cut_m(1); y <= max_cut_m(1); ++y) {
   
      for (int z = min_cut_m(2); z < min_cut(2); ++z) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }

      for (int z = max_cut(2) + 1; z <= max_cut_m(2); ++z) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }
    }

  for (int z = min_cut_m(2); z <= max_cut_m(2); ++z)
    for (int x = min_cut_m(0); x <= max_cut_m(0); ++x) {

      for (int y = min_cut_m(1); y < min_cut(1); ++y) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }

      for (int y = max_cut(1) + 1; y <= max_cut_m(1); ++y) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }
    }

  for (int y = min_cut_m(1); y <= max_cut_m(1); ++y)
    for (int z = min_cut_m(2); z <= max_cut_m(2); ++z) {

      for (int x = min_cut_m(0); x < min_cut(0); ++x) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }

      for (int x = max_cut(0) + 1; x <= max_cut_m(0); ++x) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }
    }
  Eigen::Vector3i min_id, max_id;
  posToIndex(min_pos, min_id);
  posToIndex(max_pos, max_id);

  boundIndex(min_id);
  boundIndex(max_id);

  /* reset occ and dist buffer */
  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z) {
        md_.occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;
        md_.distance_buffer_[toAddress(x, y, z)] = 10000;
      }
}

template <typename F_get_val, typename F_set_val>
void SDFMap::fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim) {
  int v[mp_.map_voxel_num_(dim)];
  double z[mp_.map_voxel_num_(dim) + 1];

  int k = start;
  v[start] = start;
  z[start] = -std::numeric_limits<double>::max();
  z[start + 1] = std::numeric_limits<double>::max();

  for (int q = start + 1; q <= end; q++) {
    k++;
    double s;

    do {
      k--;
      s = ((f_get_val(q) + q * q) - (f_get_val(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k]);
    } while (s <= z[k]);

    k++;

    v[k] = q;
    z[k] = s;
    z[k + 1] = std::numeric_limits<double>::max();
  }

  k = start;

  for (int q = start; q <= end; q++) {
    while (z[k + 1] < q) k++;
    double val = (q - v[k]) * (q - v[k]) + f_get_val(v[k]);
    f_set_val(q, val);
  }
}

void SDFMap::updateESDF3d() {
  Eigen::Vector3i min_esdf = md_.local_bound_min_;
  Eigen::Vector3i max_esdf = md_.local_bound_max_;

  /* ========== compute positive DT ========== */

  for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {
    for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
      fillESDF(
          [&](int z) {
            return md_.occupancy_buffer_inflate_[toAddress(x, y, z)] == 1 ?
                0 :
                std::numeric_limits<double>::max();
          },
          [&](int z, double val) { md_.tmp_buffer1_[toAddress(x, y, z)] = val; }, min_esdf[2],
          max_esdf[2], 2);
    }
  }

  for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {
    for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
      fillESDF([&](int y) { return md_.tmp_buffer1_[toAddress(x, y, z)]; },
               [&](int y, double val) { md_.tmp_buffer2_[toAddress(x, y, z)] = val; }, min_esdf[1],
               max_esdf[1], 1);
    }
  }

  for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
    for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
      fillESDF([&](int x) { return md_.tmp_buffer2_[toAddress(x, y, z)]; },
               [&](int x, double val) {
                 md_.distance_buffer_[toAddress(x, y, z)] = mp_.resolution_ * std::sqrt(val);
                 //  min(mp_.resolution_ * std::sqrt(val),
                 //      md_.distance_buffer_[toAddress(x, y, z)]);
               },
               min_esdf[0], max_esdf[0], 0);
    }
  }

  /* ========== compute negative distance ========== */
  for (int x = min_esdf(0); x <= max_esdf(0); ++x)
    for (int y = min_esdf(1); y <= max_esdf(1); ++y)
      for (int z = min_esdf(2); z <= max_esdf(2); ++z) {

        int idx = toAddress(x, y, z);
        if (md_.occupancy_buffer_inflate_[idx] == 0) {
          md_.occupancy_buffer_neg[idx] = 1;

        } else if (md_.occupancy_buffer_inflate_[idx] == 1) {
          md_.occupancy_buffer_neg[idx] = 0;
        } else {
          ROS_ERROR("what?");
        }
      }

  ros::Time t1, t2;

  for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {
    for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
      fillESDF(
          [&](int z) {
            return md_.occupancy_buffer_neg[x * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) +
                                            y * mp_.map_voxel_num_(2) + z] == 1 ?
                0 :
                std::numeric_limits<double>::max();
          },
          [&](int z, double val) { md_.tmp_buffer1_[toAddress(x, y, z)] = val; }, min_esdf[2],
          max_esdf[2], 2);
    }
  }

  for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {
    for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
      fillESDF([&](int y) { return md_.tmp_buffer1_[toAddress(x, y, z)]; },
               [&](int y, double val) { md_.tmp_buffer2_[toAddress(x, y, z)] = val; }, min_esdf[1],
               max_esdf[1], 1);
    }
  }

  for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
    for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
      fillESDF([&](int x) { return md_.tmp_buffer2_[toAddress(x, y, z)]; },
               [&](int x, double val) {
                 md_.distance_buffer_neg_[toAddress(x, y, z)] = mp_.resolution_ * std::sqrt(val);
               },
               min_esdf[0], max_esdf[0], 0);
    }
  }

  /* ========== combine pos and neg DT ========== */
  for (int x = min_esdf(0); x <= max_esdf(0); ++x)
    for (int y = min_esdf(1); y <= max_esdf(1); ++y)
      for (int z = min_esdf(2); z <= max_esdf(2); ++z) {

        int idx = toAddress(x, y, z);
        md_.distance_buffer_all_[idx] = md_.distance_buffer_[idx];

        if (md_.distance_buffer_neg_[idx] > 0.0)
          md_.distance_buffer_all_[idx] += (-md_.distance_buffer_neg_[idx] + mp_.resolution_);
      }
}

int SDFMap::setCacheOccupancy(Eigen::Vector3d pos, int occ) {
  if (occ != 1 && occ != 0) return INVALID_IDX;

  Eigen::Vector3i id;
  posToIndex(pos, id);
  int idx_ctns = toAddress(id);

  md_.count_hit_and_miss_[idx_ctns] += 1;

  if (md_.count_hit_and_miss_[idx_ctns] == 1) {
    md_.cache_voxel_.push(id);
  }

  if (occ == 1) md_.count_hit_[idx_ctns] += 1;

  return idx_ctns;
}



Eigen::Vector3d SDFMap::closetPointInMap(const Eigen::Vector3d& pt, const Eigen::Vector3d& camera_pt) {
  Eigen::Vector3d diff = pt - camera_pt;
  Eigen::Vector3d max_tc = mp_.map_max_boundary_ - camera_pt;
  Eigen::Vector3d min_tc = mp_.map_min_boundary_ - camera_pt;

  double min_t = 1000000;

  for (int i = 0; i < 3; ++i) {
    if (fabs(diff[i]) > 0) {

      double t1 = max_tc[i] / diff[i];
      if (t1 > 0 && t1 < min_t) min_t = t1;

      double t2 = min_tc[i] / diff[i];
      if (t2 > 0 && t2 < min_t) min_t = t2;
    }
  }

  return camera_pt + (min_t - 1e-3) * diff;
}

void SDFMap::clearAndInflateLocalMap() {
  /*clear outside local*/
  // ROS_INFO("dddddd");
  const int vec_margin = 5;
  // Eigen::Vector3i min_vec_margin = min_vec - Eigen::Vector3i(vec_margin,
  // vec_margin, vec_margin); Eigen::Vector3i max_vec_margin = max_vec +
  // Eigen::Vector3i(vec_margin, vec_margin, vec_margin);

  Eigen::Vector3i min_cut = md_.local_bound_min_ -
      Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
  Eigen::Vector3i max_cut = md_.local_bound_max_ +
      Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
  boundIndex(min_cut);
  boundIndex(max_cut);

  Eigen::Vector3i min_cut_m = min_cut - Eigen::Vector3i(vec_margin, vec_margin, vec_margin);
  Eigen::Vector3i max_cut_m = max_cut + Eigen::Vector3i(vec_margin, vec_margin, vec_margin);
  boundIndex(min_cut_m);
  boundIndex(max_cut_m);

 

  // clear data outside the local range

  for (int x = min_cut_m(0); x <= max_cut_m(0); ++x)
    for (int y = min_cut_m(1); y <= max_cut_m(1); ++y) {
   
      for (int z = min_cut_m(2); z < min_cut(2); ++z) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }

      for (int z = max_cut(2) + 1; z <= max_cut_m(2); ++z) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }
    }

  for (int z = min_cut_m(2); z <= max_cut_m(2); ++z)
    for (int x = min_cut_m(0); x <= max_cut_m(0); ++x) {

      for (int y = min_cut_m(1); y < min_cut(1); ++y) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }

      for (int y = max_cut(1) + 1; y <= max_cut_m(1); ++y) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }
    }

  for (int y = min_cut_m(1); y <= max_cut_m(1); ++y)
    for (int z = min_cut_m(2); z <= max_cut_m(2); ++z) {

      for (int x = min_cut_m(0); x < min_cut(0); ++x) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }

      for (int x = max_cut(0) + 1; x <= max_cut_m(0); ++x) {
        int idx = toAddress(x, y, z);
        md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.distance_buffer_all_[idx] = 10000;
      }
    }

  // inflate occupied voxels to compensate robot size

  int inf_step = ceil(mp_.obstacles_inflation_ / mp_.resolution_);
  int inf_step_z = 1;
  vector<Eigen::Vector3i> inf_pts(pow(2 * inf_step + 1, 3));
  // inf_pts.resize(4 * inf_step + 3);
  Eigen::Vector3i inf_pt;

  // clear outdated data
  for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
    for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y)
      for (int z = md_.local_bound_min_(2); z <= md_.local_bound_max_(2); ++z) {
        md_.occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;
      }

  // inflate obstacles
  for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
    for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y)
      for (int z = md_.local_bound_min_(2); z <= md_.local_bound_max_(2); ++z) {
        // for (int z = -inf_step_z; z <= +inf_step_z; ++z) {

        if (md_.occupancy_buffer_[toAddress(x, y, z)] > mp_.min_occupancy_log_) {
          inflatePoint(Eigen::Vector3i(x, y, z), inf_step, inf_pts);

          for (int k = 0; k < (int)inf_pts.size(); ++k) {
            inf_pt = inf_pts[k];
            int idx_inf = toAddress(inf_pt);
            if (idx_inf < 0 ||
                idx_inf >= mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2)) {
              continue;
            }
            md_.occupancy_buffer_inflate_[idx_inf] = 1;
          }
        }
      }
  
  // add virtual ceiling to limit flight height
  if (mp_.virtual_ceil_height_ > -0.5) {
    int ceil_id = floor((mp_.virtual_ceil_height_ - mp_.map_origin_(2)) * mp_.resolution_inv_);
    for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
      for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y) {
        md_.occupancy_buffer_inflate_[toAddress(x, y, ceil_id)] = 1;
        // ROS_INFO("ceiling limit x = %f, y = %f, z = %f",x,y,ceil_id);
      }
  }
 // add ground to limit flight height
 
      int ceil_id = floor((mp_.map_origin_(2)-mp_.map_origin_(2)+0.1) * mp_.resolution_inv_);
    for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
      for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y) {
        //  for (int kk =0 ; kk < 2; ++kk){
        //   int tmp = ceil_id + kk;
            md_.occupancy_buffer_inflate_[toAddress(x, y, ceil_id)] = 1;     
        // }   
      }
  
}

void SDFMap::visCallback(const ros::TimerEvent& /*event*/) {
  publishMap();
  publishMapInflate(false);
  // publishUpdateRange();
  // publishESDF();

  // publishUnknown();
  // publishDepth();
}


void SDFMap::updateESDFCallback(const ros::TimerEvent& /*event*/) {
  if (!md_.esdf_need_update_) return;
  
  /* esdf */
  ros::Time t1, t2;
  t1 = ros::Time::now();

  updateESDF3d();

  t2 = ros::Time::now();

  md_.esdf_time_ += (t2 - t1).toSec();
  md_.max_esdf_time_ = max(md_.max_esdf_time_, (t2 - t1).toSec());

  if (mp_.show_esdf_time_)
    ROS_WARN("ESDF: cur t = %lf, avg t = %lf, max t = %lf", (t2 - t1).toSec(),
             md_.esdf_time_ / md_.update_num_, md_.max_esdf_time_);



  md_.esdf_need_update_ = false;

}




Eigen::Affine3d SDFMap::transformTFToAffine3d(const tf::Transform &t) {
    Eigen::Affine3d e;
    // treat the Eigen::Affine as a 4x4 matrix:
    for (int i = 0; i < 3; i++) {
        e.matrix()(i, 3) = t.getOrigin()[i]; //copy the origin from tf to Eigen
        for (int j = 0; j < 3; j++) {
            e.matrix()(i, j) = t.getBasis()[i][j]; //and copy 3x3 rotation matrix
        }
    }
    // Fill in identity in last row
    for (int col = 0; col < 3; col++)
        e.matrix()(3, col) = 0;
    e.matrix()(3, 3) = 1;
    return e;
}



void SDFMap::publishMap() {

  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = md_.local_bound_min_;
  Eigen::Vector3i max_cut = md_.local_bound_max_;

  int lmm = mp_.local_map_margin_ / 2;
  min_cut -= Eigen::Vector3i(lmm, lmm, lmm);
  max_cut += Eigen::Vector3i(lmm, lmm, lmm);

  boundIndex(min_cut);
  boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z) {
        if (md_.occupancy_buffer_inflate_[toAddress(x, y, z)] == 0) continue;

        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);
        if (pos(2) > mp_.visualization_truncate_height_) continue;

        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = pos(2);
        cloud.push_back(pt);
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  map_pub_.publish(cloud_msg);
}

void SDFMap::publishMapInflate(bool all_info) {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = md_.local_bound_min_;
  Eigen::Vector3i max_cut = md_.local_bound_max_;

  if (all_info) {
    int lmm = mp_.local_map_margin_;
    min_cut -= Eigen::Vector3i(lmm, lmm, lmm);
    max_cut += Eigen::Vector3i(lmm, lmm, lmm);
  }

  boundIndex(min_cut);
  boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z) {
        if (md_.occupancy_buffer_inflate_[toAddress(x, y, z)] == 0) continue;

        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);
        if (pos(2) > mp_.visualization_truncate_height_) continue;

        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = pos(2);
        cloud.push_back(pt);
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  map_inf_pub_.publish(cloud_msg);

  // ROS_INFO("pub map");
}





void SDFMap::publishUpdateRange() {
  Eigen::Vector3d esdf_min_pos, esdf_max_pos, cube_pos, cube_scale;
  visualization_msgs::Marker mk;
  indexToPos(md_.local_bound_min_, esdf_min_pos);
  indexToPos(md_.local_bound_max_, esdf_max_pos);

  cube_pos = 0.5 * (esdf_min_pos + esdf_max_pos);
  cube_scale = esdf_max_pos - esdf_min_pos;
  mk.header.frame_id = mp_.frame_id_;
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::CUBE;
  mk.action = visualization_msgs::Marker::ADD;
  mk.id = 0;

  mk.pose.position.x = cube_pos(0);
  mk.pose.position.y = cube_pos(1);
  mk.pose.position.z = cube_pos(2);

  mk.scale.x = cube_scale(0);
  mk.scale.y = cube_scale(1);
  mk.scale.z = cube_scale(2);

  mk.color.a = 0.3;
  mk.color.r = 1.0;
  mk.color.g = 0.0;
  mk.color.b = 0.0;

  mk.pose.orientation.w = 1.0;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;

  update_range_pub_.publish(mk);
}

void SDFMap::publishESDF() {
  double dist;
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZI pt;

  const double min_dist = 0.0;
  const double max_dist = 3.0;

  Eigen::Vector3i min_cut = md_.local_bound_min_ -
      Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
  Eigen::Vector3i max_cut = md_.local_bound_max_ +
      Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
  boundIndex(min_cut);
  boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y) {

      Eigen::Vector3d pos;
      indexToPos(Eigen::Vector3i(x, y, 1), pos);
      pos(2) = mp_.esdf_slice_height_;

      dist = getDistance(pos);
      dist = min(dist, max_dist);
      dist = max(dist, min_dist);

      pt.x = pos(0);
      pt.y = pos(1);
      pt.z = -0.2;
      pt.intensity = (dist - min_dist) / (max_dist - min_dist);
      cloud.push_back(pt);
    }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);

  esdf_pub_.publish(cloud_msg);

  // ROS_INFO("pub esdf");
}

void SDFMap::getSliceESDF(const double height, const double res, const Eigen::Vector4d& range,
                          vector<Eigen::Vector3d>& slice, vector<Eigen::Vector3d>& grad, int sign) {
  double dist;
  Eigen::Vector3d gd;
  for (double x = range(0); x <= range(1); x += res)
    for (double y = range(2); y <= range(3); y += res) {

      dist = this->getDistWithGradTrilinear(Eigen::Vector3d(x, y, height), gd);
      slice.push_back(Eigen::Vector3d(x, y, dist));
      grad.push_back(gd);
    }
}

void SDFMap::checkDist() {
  for (int x = 0; x < mp_.map_voxel_num_(0); ++x)
    for (int y = 0; y < mp_.map_voxel_num_(1); ++y)
      for (int z = 0; z < mp_.map_voxel_num_(2); ++z) {
        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);

        Eigen::Vector3d grad;
        double dist = getDistWithGradTrilinear(pos, grad);

        if (fabs(dist) > 10.0) {
        }
      }
}

bool SDFMap::odomValid() { return md_.has_odom_; }

bool SDFMap::hasDepthObservation() { return md_.has_first_depth_; }

double SDFMap::getResolution() { return mp_.resolution_; }

Eigen::Vector3d SDFMap::getOrigin() { return mp_.map_origin_; }

int SDFMap::getVoxelNum() {
  return mp_.map_voxel_num_[0] * mp_.map_voxel_num_[1] * mp_.map_voxel_num_[2];
}

void SDFMap::getRegion(Eigen::Vector3d& ori, Eigen::Vector3d& size) {
  ori = mp_.map_origin_, size = mp_.map_size_;
}

void SDFMap::getSurroundPts(const Eigen::Vector3d& pos, Eigen::Vector3d pts[2][2][2],
                            Eigen::Vector3d& diff) {
  if (!isInMap(pos)) {
    // cout << "pos invalid for interpolation." << endl;
  }

  /* interpolation position */
  Eigen::Vector3d pos_m = pos - 0.5 * mp_.resolution_ * Eigen::Vector3d::Ones();
  Eigen::Vector3i idx;
  Eigen::Vector3d idx_pos;

  posToIndex(pos_m, idx);
  indexToPos(idx, idx_pos);
  diff = (pos - idx_pos) * mp_.resolution_inv_;

  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        Eigen::Vector3i current_idx = idx + Eigen::Vector3i(x, y, z);
        Eigen::Vector3d current_pos;
        indexToPos(current_idx, current_pos);
        pts[x][y][z] = current_pos;
      }
    }
  }
}

void SDFMap::odomLaserCloudCallback(const nav_msgs::OdometryConstPtr& odom,   
                                    const sensor_msgs::LaserScanConstPtr &scan_in, 
                                    const sensor_msgs::PointCloud2ConstPtr& img,
                                    const sensor_msgs::RangeConstPtr& oneDLidar) {
  md_.has_first_depth_ = true;
  md_.has_odom_ = true;
  md_.has_cloud_ = true;


  if (isnan(md_.camera_pos_(0)) || isnan(md_.camera_pos_(1)) || isnan(md_.camera_pos_(2))) return;


  odom_msg = *odom;
  md_.camera_pos_(0) = odom->pose.pose.position.x;
  md_.camera_pos_(1) = odom->pose.pose.position.y;
  md_.camera_pos_(2) = odom->pose.pose.position.z;
  // md_.camera_q_ = Eigen::Quaterniond(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x,
  //                                     odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);
 
 Eigen::Vector3d oneDLidar_global;
  oneDLidar_global(0) = md_.camera_pos_(0);
  oneDLidar_global(1) = md_.camera_pos_(1);  
  if(oneDLidar->range < oneDLidar->max_range || oneDLidar->range > oneDLidar->min_range)
    {      
      oneDLidar_global(2) = md_.camera_pos_(2) - oneDLidar->range;
    }else{
       oneDLidar_global(2) = 0.0;
    }
 
    tf::Quaternion q_(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x,odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);
    tf::Matrix3x3 m(q_);            
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);        
    current_yaw = yaw;

 
  int inf_step = ceil(mp_.obstacles_inflation_ / mp_.resolution_);
  int inf_step_z = 1;

  //////////////////////////////////////////////
  /////////////////////////////////////////////
  ////////////////////////////////////////////

    Eigen::Vector3d oned_inf;
    Eigen::Vector3i oned_pt;   
            for (int x = -inf_step; x <= inf_step; ++x)
              for (int y = -inf_step; y <= inf_step; ++y)
                for (int z = -1; z <= 0; ++z) {

                  oned_inf(0) = oneDLidar_global(0) + x * mp_.resolution_;
                  oned_inf(1) = oneDLidar_global(1) + y * mp_.resolution_;
                  oned_inf(2) = oneDLidar_global(2) + z * mp_.resolution_;

                  

                  posToIndex(oned_inf, oned_pt);
                  if (!isInMap(oned_pt)) continue;
                                      
                  int idx_inf = toAddress(oned_pt);

                  md_.occupancy_buffer_inflate_[idx_inf] = 1;
                }
  /////////////////////////////////////////
  //////////////////////////////////////

   if(!lidar_sub_done){     
    laser_in_back_left.angle_min = -3.141958;
    laser_in_back_left.angle_max = scan_in->angle_min;
    laser_in_back_left.angle_increment = scan_in->angle_increment;
    laser_in_back_left.time_increment = scan_in->time_increment;
    laser_in_back_left.scan_time = scan_in->scan_time;
    laser_in_back_left.range_min = scan_in->range_min;
    laser_in_back_left.range_max = scan_in->range_max;
    for (int i = 0; i < (laser_in_back_left.angle_max - laser_in_back_left.angle_min)/laser_in_back_left.angle_increment+1; i++){
      laser_in_back_left.ranges.push_back(2.0);
      laser_in_back_left.intensities.push_back(1000.0);
    }    
    laser_in_back_right.angle_min = scan_in->angle_max;
    laser_in_back_right.angle_max = 3.141958;
    laser_in_back_right.angle_increment = scan_in->angle_increment;
    laser_in_back_right.time_increment = scan_in->time_increment;
    laser_in_back_right.scan_time = scan_in->scan_time;
    laser_in_back_right.range_min = scan_in->range_min;
    laser_in_back_right.range_max = scan_in->range_max;
      for (int i = 0; i < (laser_in_back_right.angle_max - laser_in_back_right.angle_min)/laser_in_back_right.angle_increment+1; i++){
        laser_in_back_right.ranges.push_back(2.0);
        laser_in_back_right.intensities.push_back(1000.0);
      }
  }
  laser_in_back_left.header = scan_in->header;
  laser_in_back_right.header = scan_in->header;

  lidar_data =*scan_in;
  lidar_frame_ = scan_in->header.frame_id;
  lidar_sub_done =true;


  pcl::PointCloud<pcl::PointXYZ> latest_cloud;
  pcl::PointCloud<pcl::PointXYZ>::Ptr latest_cloud_tmp (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*img, *latest_cloud_tmp);

  ////////////  Downsample pcl///////////////////
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud (latest_cloud_tmp);             
        sor.setLeafSize (0.2f, 0.2f, 0.2f); //leaf size
        sor.filter (*latest_cloud_tmp);       





  //////////////////////////////////////////////////////////////////////////////////////
  // transform point cloud from ** depth camera ** to world frame
  //////////////////////////////////////////////////////////////////////////////////////
  tf::StampedTransform robot_to_world;
  try {  
    tf_listener_.waitForTransform(world_frame_, robot_frame_,odom_msg.header.stamp,ros::Duration(0.2));
    tf_listener_.lookupTransform(world_frame_, robot_frame_, odom_msg.header.stamp,robot_to_world);
  } catch (tf::TransformException& ex) {
    ROS_ERROR_STREAM(
        "Error getting TF transform from sensor data: " << ex.what());
    return;
  }  
  tf::Transform transform_tmp(robot_to_world.getBasis(), robot_to_world.getOrigin());
  Eigen::Affine3d affine_transform_tmp = transformTFToAffine3d(transform_tmp);
  pcl::transformPointCloud(*latest_cloud_tmp, *latest_cloud_tmp, affine_transform_tmp);
  latest_cloud = *latest_cloud_tmp;


  if (latest_cloud.points.size() == 0) return;

  if (isnan(md_.camera_pos_(0)) || isnan(md_.camera_pos_(1)) || isnan(md_.camera_pos_(2))) return;

  this->resetBuffer(md_.camera_pos_ - mp_.local_update_range_,
                    md_.camera_pos_ + mp_.local_update_range_);

  pcl::PointXYZ pt;
  Eigen::Vector3d p3d, p3d_inf;

  
  inf_step_z = 1;

  double max_x, max_y, max_z, min_x, min_y, min_z;

  min_x = mp_.map_max_boundary_(0);
  min_y = mp_.map_max_boundary_(1);
  min_z = mp_.map_max_boundary_(2);

  max_x = mp_.map_min_boundary_(0);
  max_y = mp_.map_min_boundary_(1);
  max_z = mp_.map_min_boundary_(2);


 if (mp_.virtual_ceil_height_ > -0.5) {
    int ceil_id = floor((mp_.virtual_ceil_height_ - mp_.map_origin_(2)) * mp_.resolution_inv_);
    for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
      for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y) {
        md_.occupancy_buffer_inflate_[toAddress(x, y, ceil_id)] = 1;        
      }
  }
 // add ground to limit flight height
      int ceil_id = floor(( mp_.map_origin_(2) - mp_.map_origin_(2)) * mp_.resolution_inv_);
    for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
      for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y) {
        //  for (int kk =0 ; kk < 2; ++kk){
        //   int tmp = ceil_id + kk;
            md_.occupancy_buffer_inflate_[toAddress(x, y, ceil_id)] = 1;     
        // }   
      }


  for (size_t i = 0; i < latest_cloud.points.size(); ++i) {
    pt = latest_cloud.points[i];    
    double dist_to_pt = sqrt( pow((pt.x-md_.camera_pos_[0]),2) 
              +pow((pt.y-md_.camera_pos_[1]),2) 
              +pow((pt.z-md_.camera_pos_[2]),2) ); 

    if(  dist_to_pt < mp_.min_ray_length_ || dist_to_pt >  mp_.max_ray_length_){
        continue;
      }

      
    p3d(0) = pt.x, p3d(1) = pt.y, p3d(2) = pt.z;

    
    /* point inside update range */
    Eigen::Vector3d devi = p3d - md_.camera_pos_;
    // Eigen::Vector3d devi = p3d ;
    Eigen::Vector3i inf_pt;

    if (fabs(devi(0)) < mp_.local_update_range_(0) && fabs(devi(1)) < mp_.local_update_range_(1) &&
        fabs(devi(2)) < mp_.local_update_range_(2)) {

      /* inflate the point */
      for (int x = -inf_step; x <= inf_step; ++x)
        for (int y = -inf_step; y <= inf_step; ++y)
          for (int z = -inf_step_z; z <= inf_step_z; ++z) {

            p3d_inf(0) = pt.x + x * mp_.resolution_;
            p3d_inf(1) = pt.y + y * mp_.resolution_;
            p3d_inf(2) = pt.z + z * mp_.resolution_;

            max_x = max(max_x, p3d_inf(0));
            max_y = max(max_y, p3d_inf(1));
            max_z = max(max_z, p3d_inf(2));

            min_x = min(min_x, p3d_inf(0));
            min_y = min(min_y, p3d_inf(1));
            min_z = min(min_z, p3d_inf(2));

            posToIndex(p3d_inf, inf_pt);

            if (!isInMap(inf_pt)) continue;

            int idx_inf = toAddress(inf_pt);

            md_.occupancy_buffer_inflate_[idx_inf] = 1;
          }
    }
  }

  double min_x_devi = 0.0;
  double max_x_devi = 0.0;
  double min_y_devi = 0.0;
  double max_y_devi = 0.0;  //////////////////////////////////////////////////////////////////////////////////////
  // tranfrom point cloud from ** Lidar ** to world frame 
  //////////////////////////////////////////////////////////////////////////////////////  
try {    
        projector_.projectLaser (laser_in_back_left,pcd_from_lidar_left_back);        
        projector_.projectLaser (laser_in_back_right,pcd_from_lidar_right_back);
        tf_listener_.waitForTransform(scan_in->header.frame_id,
                                      "/base_link",
                                      scan_in->header.stamp+ ros::Duration().fromSec(scan_in->ranges.size()*scan_in->time_increment),
                                      ros::Duration(0.3));
        
        projector_.transformLaserScanToPointCloud("/base_link",*scan_in,pcd_from_lidar,tf_listener_);
     
  if(lidar_sub_done){
    pcl::PointCloud<pcl::PointXYZ> latest_lidar_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cloud_tmp (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(pcd_from_lidar, *lidar_cloud_tmp);

    pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cloud_tmp_left_back (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(pcd_from_lidar_left_back, *lidar_cloud_tmp_left_back);
  
    pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cloud_tmp_right_back (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(pcd_from_lidar_right_back, *lidar_cloud_tmp_right_back);

    *lidar_cloud_tmp += (*lidar_cloud_tmp_left_back);
    *lidar_cloud_tmp += (*lidar_cloud_tmp_right_back);

  pcl::PointCloud<pcl::PointXYZ>::Ptr p_obstacles(new pcl::PointCloud<pcl::PointXYZ>);

//////////////////////////////// FILTER points inside of field of view ///////////////////
  

    tf::StampedTransform lidar_to_world;
    try {
      tf_listener_.waitForTransform(world_frame_, lidar_frame_,odom_msg.header.stamp,ros::Duration(0.3));
      tf_listener_.lookupTransform(world_frame_, lidar_frame_, odom_msg.header.stamp,lidar_to_world);
    } catch (tf::TransformException& ex) {
      ROS_ERROR_STREAM(
          "Error getting TF transform from sensor data: " << ex.what());
      return;
    }  
    tf::Transform transform_lidar_tmp(lidar_to_world.getBasis(), lidar_to_world.getOrigin());
    Eigen::Affine3d affine_transform_lidar_tmp = transformTFToAffine3d(transform_lidar_tmp);
    pcl::transformPointCloud(*lidar_cloud_tmp, *lidar_cloud_tmp, affine_transform_lidar_tmp);
    latest_lidar_cloud = *lidar_cloud_tmp;

    inf_step = ceil(mp_.obstacles_inflation_ / mp_.resolution_);
    bool devi_init= false;
    for (size_t i = 0; i < latest_lidar_cloud.points.size(); ++i) {
      pt = latest_lidar_cloud.points[i]; 
////////////////////////////////////////////
      //// check if the points inside of camera field of view 
////////////////////////////////////////////
 double angle_from_point = atan2((pt.y - md_.camera_pos_(1)),( pt.x- md_.camera_pos_(0)));        
          double tmp_ = current_yaw-angle_from_point;
          truncateYaw(tmp_);      
          double tmp_diff_angle = fabs(tmp_);     
    
     
////////////////////////////////////////////
         p3d(0) = pt.x, p3d(1) = pt.y, p3d(2) = pt.z;            
          /* point inside update range */
          Eigen::Vector3d devi = p3d - md_.camera_pos_;
          
               
             double global_direction_angle_tmp_ = global_direction-angle_from_point;
          truncateYaw(global_direction_angle_tmp_);      
          double global_direction_tmp_diff_angle = fabs(global_direction_angle_tmp_);       
          
          if( global_direction_tmp_diff_angle < PI/2){
              if(!devi_init){
                min_x_devi = devi(0);
                max_x_devi = devi(0);
                min_y_devi = devi(1);
                max_y_devi = devi(1);
                devi_init = true;
              }            
              if( min_x_devi >= devi(0)){
                min_x_devi = devi(0);
              }
              if(max_x_devi <= devi(0)){
                max_x_devi = devi(0);
              }
              if( min_y_devi >= devi(1)){
                min_y_devi = devi(1);
              }
              if(max_y_devi <= devi(1)){
                max_y_devi = devi(1);
              }
          }       
        

          Eigen::Vector3i inf_pt;          
          /////////////////////////
        if (tmp_diff_angle < 0.3419f){     
              if (fabs(devi(0)) < 0.6 && fabs(devi(1)) < 0.6 ) continue;
              // Eigen::Vector3d devi = p3d ;
              // if (fabs(devi(0)) < mp_.local_update_range_(0)-mp_.obstacles_inflation_-mp_.resolution_ && fabs(devi(1)) < mp_.local_update_range_(1)-mp_.obstacles_inflation_ -mp_.resolution_&&
              if (fabs(devi(0)) < mp_.local_update_range_(0) && fabs(devi(1)) < mp_.local_update_range_(1) &&
                  fabs(devi(2)) < mp_.local_update_range_(2)) {          
                /* inflate the point */
                for (int x = -inf_step; x <= inf_step; ++x)
                  for (int y = -inf_step; y <= inf_step; ++y)
                    for (int z = -inf_step_z; z <= inf_step_z; ++z) {

                      p3d_inf(0) = pt.x + x * mp_.resolution_;
                      p3d_inf(1) = pt.y + y * mp_.resolution_;
                      p3d_inf(2) = pt.z + z * mp_.resolution_;

                      max_x = max(max_x, p3d_inf(0));
                      max_y = max(max_y, p3d_inf(1));
                      max_z = max(max_z, p3d_inf(2));

                      min_x = min(min_x, p3d_inf(0));
                      min_y = min(min_y, p3d_inf(1));
                      min_z = min(min_z, p3d_inf(2));

                      posToIndex(p3d_inf, inf_pt);

                      if (!isInMap(inf_pt)) continue;
                      //skip points inside of robot dimension 
                      Eigen::Vector3d devi_local = p3d_inf-md_.camera_pos_;     
                      if (fabs(devi_local(0)) < 0.6 && fabs(devi_local(1)) < 0.6 ) continue;

                      int idx_inf = toAddress(inf_pt);
                      md_.occupancy_buffer_inflate_[idx_inf] = 1;
                    }
                }

          } else{
          //// /// laser data over the field of view (create virtual wall)
                for (double kk=-(pt.z)-1; kk<(4-pt.z) ; kk=kk+mp_.resolution_){      
                    p3d(0) = pt.x, p3d(1) = pt.y, p3d(2) = pt.z+kk;            
                    /* point inside update range */
                    Eigen::Vector3d devi = p3d - md_.camera_pos_;
                    if (fabs(devi(0)) < 0.4 && fabs(devi(1)) < 0.4 ) continue;
                    // Eigen::Vector3d devi = p3d ;
                    Eigen::Vector3i inf_pt;
                    // if (fabs(devi(0)) < mp_.local_update_range_(0)-mp_.obstacles_inflation_-mp_.resolution_ && fabs(devi(1)) < mp_.local_update_range_(1)-mp_.obstacles_inflation_ -mp_.resolution_&&
                    if (fabs(devi(0)) < mp_.local_update_range_(0) && fabs(devi(1)) < mp_.local_update_range_(1)&&
                        fabs(devi(2)) < mp_.local_update_range_(2)) {
                    
                      /* inflate the point */
                      for (int x = -inf_step; x <= inf_step; ++x)
                        for (int y = -inf_step; y <= inf_step; ++y)
                          for (int z = -inf_step; z <= inf_step; ++z) {

                            p3d_inf(0) = pt.x + x * mp_.resolution_;
                            p3d_inf(1) = pt.y + y * mp_.resolution_;
                            p3d_inf(2) = pt.z +kk+ z * mp_.resolution_;

                            max_x = max(max_x, p3d_inf(0));
                            max_y = max(max_y, p3d_inf(1));
                            max_z = max(max_z, p3d_inf(2));

                            min_x = min(min_x, p3d_inf(0));
                            min_y = min(min_y, p3d_inf(1));
                            min_z = min(min_z, p3d_inf(2));

                            posToIndex(p3d_inf, inf_pt);

                            if (!isInMap(inf_pt)) continue;
                            //skip points inside of robot dimension 
                            Eigen::Vector3d devi_local =  p3d_inf-md_.camera_pos_;      
                            if (fabs(devi_local(0)) < 0.6 && fabs(devi_local(1)) < 0.6 && fabs(devi_local(2))< 0.6) continue;

                            int idx_inf = toAddress(inf_pt);

                            md_.occupancy_buffer_inflate_[idx_inf] = 1;
                          }
                      }
                  }
              }
        } 
      } 
  

      if( global_direction >= -PI/4.0 && global_direction <= PI/4.0){
           // current direction = go to +x direction         
            if(fabs(max_y_devi) > fabs(min_y_devi)){
                if(fabs(max_x_devi) < fabs(max_y_devi)){
                    global_direction = PI/2.0; // go to +y direction 
                    global_direction_max_distance = max_y_devi;
                  }
            }else{
              if(fabs(max_x_devi) < fabs(min_y_devi)){
                   global_direction = -PI/2.0; // go to -y direction 
                   global_direction_max_distance = min_y_devi;
              }
            }
      }else if(global_direction >= -PI/4.0+PI/2.0 && global_direction <= PI/4.0+PI/2.0){
        // current direction = go to +y direction 
          if(fabs(max_x_devi) > fabs(min_x_devi)){          
                if(fabs(max_y_devi) < fabs(max_x_devi)){
                    global_direction = 0.0; // go to +x direction 
                    global_direction_max_distance = max_x_devi;
                  }
            }else{
              if(fabs(max_y_devi) < fabs(min_x_devi)){
                   global_direction = -PI; // go to -x direction
                    global_direction_max_distance = min_x_devi;
              }
            }
      }else if(global_direction >= -PI-PI/4.0 && global_direction <= -PI+PI/4.0){
        // current direction = go to -x direction 
          if(fabs(max_y_devi) > fabs(min_y_devi)){          
                if(fabs(min_x_devi) < fabs(max_y_devi)){
                    global_direction = PI/2.0; // go to +y direction 
                     global_direction_max_distance = max_y_devi;
                  }
            }else{
              if(fabs(min_x_devi) < fabs(min_y_devi)){
                   global_direction = -PI/2.0; // go to -y direction
                    global_direction_max_distance = min_y_devi;
              }
            }
      }else if(global_direction >= -PI/4.0-PI/2.0 && global_direction <= PI/4.0-PI/2.0){
        // current direction = go to -y direction 
          if(fabs(max_x_devi) > fabs(min_x_devi)){          
                  if(fabs(min_y_devi) < fabs(max_x_devi)){
                      global_direction = 0.0; // go to +x direction 
                      global_direction_max_distance = max_x_devi;
                    }
              }else{
                if(fabs(min_y_devi) < fabs(min_x_devi)){
                    global_direction = -PI; // go to -x direction
                     global_direction_max_distance = min_x_devi;
                }
              }
      }
    std_msgs::Float32 goal_dir_;
    goal_dir_.data = global_direction;
    goal_direction_pub_.publish(goal_dir_);
    


  
 } catch (tf::TransformException& ex) {
      ROS_ERROR_STREAM(
          "Error getting TF transform from sensor sdata: " << ex.what());
          lidar_sub_done =false;
      return;
    }


  min_x = min(min_x, md_.camera_pos_(0));
  min_y = min(min_y, md_.camera_pos_(1));
  min_z = min(min_z, md_.camera_pos_(2));

  max_x = max(max_x, md_.camera_pos_(0));
  max_y = max(max_y, md_.camera_pos_(1));
  max_z = max(max_z, md_.camera_pos_(2));

  max_z = max(max_z, mp_.ground_height_);

  posToIndex(Eigen::Vector3d(max_x, max_y, max_z), md_.local_bound_max_);
  posToIndex(Eigen::Vector3d(min_x, min_y, min_z), md_.local_bound_min_);

  boundIndex(md_.local_bound_min_);
  boundIndex(md_.local_bound_max_);

  md_.esdf_need_update_ = true;


}

double SDFMap::get_global_direction(){
  return global_direction;
}

double SDFMap::get_global_direction_max_distance(){
  return global_direction_max_distance;
}


void SDFMap::depthCallback(const sensor_msgs::ImageConstPtr& img) {
  std::cout << "depth: " << img->header.stamp << std::endl;
}

void SDFMap::poseCallback(const geometry_msgs::PoseStampedConstPtr& pose) {
  std::cout << "pose: " << pose->header.stamp << std::endl;

  md_.camera_pos_(0) = pose->pose.position.x;
  md_.camera_pos_(1) = pose->pose.position.y;
  md_.camera_pos_(2) = pose->pose.position.z;
}

// SDFMap
