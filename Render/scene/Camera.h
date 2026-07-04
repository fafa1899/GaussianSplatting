#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

class Camera {
 public:
  int width = 1280;
  int height = 720;
  float fovYDegrees = 60.0f;
  float zNear = 0.1f;
  float zFar = 1000.0f;

  Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();

  static Camera CreateLookAt(int width, int height, float fovYDegrees,
                             float zNear, float zFar,
                             const Eigen::Vector3f& eye,
                             const Eigen::Vector3f& target,
                             const Eigen::Vector3f& up);
};