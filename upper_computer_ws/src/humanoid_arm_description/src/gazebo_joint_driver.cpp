#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace humanoid_arm_description
{

class GazeboJointDriver final : public gazebo::ModelPlugin
{
public:
  GazeboJointDriver() = default;

  ~GazeboJointDriver() override
  {
    update_connection_.reset();
  }

  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = std::move(model);
    node_ = gazebo_ros::Node::Get(sdf);

    direct_mode_ = sdf->Get<bool>("direct_mode", false).first;
    publish_rate_hz_ = sdf->Get<double>("publish_rate_hz", 60.0).first;
    if (!std::isfinite(publish_rate_hz_) || publish_rate_hz_ <= 0.0) {
      throw std::runtime_error("publish_rate_hz must be finite and greater than zero");
    }

    const std::array<double, kJointCount> default_min{{
      -1.483529864, -0.174532925, -1.745329252, -0.698131701}};
    const std::array<double, kJointCount> default_max{{
      3.054326191, 2.617993878, 1.745329252, 1.745329252}};
    const std::array<double, kJointCount> default_speed{{
      3.141592654, 3.141592654, 3.141592654, 3.141592654}};

    for (std::size_t index = 0; index < kJointCount; ++index) {
      const std::string suffix = std::to_string(index + 1U);
      joint_names_[index] = "joint_" + suffix;
      joints_[index] = model_->GetJoint(joint_names_[index]);
      if (!joints_[index]) {
        throw std::runtime_error("model is missing " + joint_names_[index]);
      }
      lower_limits_[index] = sdf->Get<double>(
        "joint_" + suffix + "_min", default_min[index]).first;
      upper_limits_[index] = sdf->Get<double>(
        "joint_" + suffix + "_max", default_max[index]).first;
      max_speeds_[index] = sdf->Get<double>(
        "joint_" + suffix + "_max_speed", default_speed[index]).first;
      if (!std::isfinite(lower_limits_[index]) ||
        !std::isfinite(upper_limits_[index]) ||
        !std::isfinite(max_speeds_[index]) ||
        lower_limits_[index] > upper_limits_[index] || max_speeds_[index] <= 0.0)
      {
        throw std::runtime_error("invalid limits for " + joint_names_[index]);
      }
      targets_[index] = joints_[index]->Position(0);
    }

    const auto command_topic = sdf->Get<std::string>(
      "command_topic", "/kinematics/joint_command").first;
    const auto state_topic = sdf->Get<std::string>(
      "state_topic", "/kinematics/joint_state").first;
    command_sub_ = node_->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      command_topic, rclcpp::QoS(5),
      std::bind(&GazeboJointDriver::OnCommand, this, std::placeholders::_1));
    state_pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(
      state_topic, rclcpp::QoS(5));

    last_update_time_ = model_->GetWorld()->SimTime();
    last_publish_time_ = last_update_time_;
    update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&GazeboJointDriver::OnUpdate, this));

    RCLCPP_INFO(
      node_->get_logger(), "Gazebo arm driver ready (%s mode)",
      direct_mode_ ? "direct" : "interpolated");
  }

private:
  static constexpr std::size_t kJointCount = 4U;

  void OnCommand(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
  {
    if (msg->points.empty()) {
      Reject("trajectory contains no points");
      return;
    }

    std::unordered_map<std::string, std::size_t> indices;
    for (std::size_t index = 0; index < msg->joint_names.size(); ++index) {
      if (!indices.emplace(msg->joint_names[index], index).second) {
        Reject("trajectory contains duplicate joint names");
        return;
      }
    }

    const auto & positions = msg->points.back().positions;
    const auto & velocities = msg->points.back().velocities;
    std::array<double, kJointCount> requested{};
    for (std::size_t index = 0; index < kJointCount; ++index) {
      const auto entry = indices.find(joint_names_[index]);
      if (entry == indices.end() || entry->second >= positions.size()) {
        Reject("trajectory is missing " + joint_names_[index]);
        return;
      }
      const double value = positions[entry->second];
      if (!std::isfinite(value)) {
        Reject(joint_names_[index] + " target is not finite");
        return;
      }
      if (value < lower_limits_[index] || value > upper_limits_[index]) {
        Reject(joint_names_[index] + " target exceeds configured limits");
        return;
      }
      if (!velocities.empty()) {
        if (entry->second >= velocities.size() ||
          !std::isfinite(velocities[entry->second]) ||
          std::abs(velocities[entry->second]) > max_speeds_[index])
        {
          Reject(joint_names_[index] + " velocity exceeds configured limits");
          return;
        }
      }
      requested[index] = value;
    }

    std::lock_guard<std::mutex> lock(target_mutex_);
    targets_ = requested;
  }

  void OnUpdate()
  {
    const auto sim_time = model_->GetWorld()->SimTime();
    const double dt = (sim_time - last_update_time_).Double();
    last_update_time_ = sim_time;
    if (!std::isfinite(dt) || dt <= 0.0) {
      return;
    }

    std::array<double, kJointCount> targets;
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      targets = targets_;
    }

    for (std::size_t index = 0; index < kJointCount; ++index) {
      const double current = joints_[index]->Position(0);
      const double error = targets[index] - current;
      const double step = direct_mode_ ? error : std::max(
        -max_speeds_[index] * dt,
        std::min(max_speeds_[index] * dt, error));
      joints_[index]->SetPosition(0, current + step, true);
    }

    if ((sim_time - last_publish_time_).Double() >= 1.0 / publish_rate_hz_) {
      PublishState();
      last_publish_time_ = sim_time;
    }
  }

  void PublishState()
  {
    sensor_msgs::msg::JointState msg;
    msg.header.stamp = node_->get_clock()->now();
    msg.header.frame_id = "base";
    msg.name.assign(joint_names_.begin(), joint_names_.end());
    msg.position.reserve(kJointCount);
    msg.velocity.reserve(kJointCount);
    for (const auto & joint : joints_) {
      msg.position.push_back(joint->Position(0));
      msg.velocity.push_back(joint->GetVelocity(0));
    }
    state_pub_->publish(msg);
  }

  void Reject(const std::string & reason) const
  {
    RCLCPP_WARN(node_->get_logger(), "Rejected joint command: %s", reason.c_str());
  }

  gazebo::physics::ModelPtr model_;
  gazebo_ros::Node::SharedPtr node_;
  std::array<std::string, kJointCount> joint_names_{};
  std::array<gazebo::physics::JointPtr, kJointCount> joints_{};
  std::array<double, kJointCount> lower_limits_{};
  std::array<double, kJointCount> upper_limits_{};
  std::array<double, kJointCount> max_speeds_{};
  std::array<double, kJointCount> targets_{};
  std::mutex target_mutex_;
  bool direct_mode_{false};
  double publish_rate_hz_{60.0};
  gazebo::common::Time last_update_time_;
  gazebo::common::Time last_publish_time_;
  gazebo::event::ConnectionPtr update_connection_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr command_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr state_pub_;
};

GZ_REGISTER_MODEL_PLUGIN(GazeboJointDriver)

}  // namespace humanoid_arm_description
