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



#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include <plan_manage/kino_replan_fsm.h>
#include <plan_manage/topo_replan_fsm.h>

#include <plan_manage/backward.hpp>
namespace backward {
backward::SignalHandling sh;
}

using namespace fast_planner;

int main(int argc, char** argv) {
  ros::init(argc, argv, "fast_planner_node");
  ros::NodeHandle nh("~");

  ros::NodeHandle map_nh;  
  ros::NodeHandle service_nh;  

    ros::CallbackQueue callback_queue_map;
    map_nh.setCallbackQueue(&callback_queue_map);

    std::thread spinner_thread_map([&callback_queue_map]() {
    ros::SingleThreadedSpinner spinner_map;
    spinner_map.spin(&callback_queue_map);
    });
///////////////
    ros::CallbackQueue callback_queue_service;
    service_nh.setCallbackQueue(&callback_queue_service);

    std::thread spinner_thread_service([&callback_queue_service]() {
    ros::SingleThreadedSpinner spinner_service;
    spinner_service.spin(&callback_queue_service);
    });

    
   

  int planner;
  nh.param("planner_node/planner", planner, -1);

  TopoReplanFSM topo_replan;
  KinoReplanFSM kino_replan;

  if (planner == 1) {
    kino_replan.init(nh,map_nh,service_nh);
  } else if (planner == 2) {
    topo_replan.init(nh,map_nh);
  }

  ros::Duration(1.0).sleep();
  ros::spin();
  spinner_thread_map.join();
  spinner_thread_service.join();

  return 0;
}
