#include "math/Transform.h"

#include <Eigen/Geometry>
#include <cmath>

Eigen::Matrix4f MakeLookAt(const Eigen::Vector3f& eye,
                           const Eigen::Vector3f& target,
                           const Eigen::Vector3f& up) {
  const Eigen::Vector3f forward = (target - eye).normalized();
  const Eigen::Vector3f right = forward.cross(up).normalized();
  const Eigen::Vector3f cameraUp = right.cross(forward).normalized();

  Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
  view(0, 0) = right.x();
  view(0, 1) = right.y();
  view(0, 2) = right.z();
  view(0, 3) = -right.dot(eye);

  view(1, 0) = cameraUp.x();
  view(1, 1) = cameraUp.y();
  view(1, 2) = cameraUp.z();
  view(1, 3) = -cameraUp.dot(eye);

  view(2, 0) = -forward.x();
  view(2, 1) = -forward.y();
  view(2, 2) = -forward.z();
  view(2, 3) = forward.dot(eye);

  return view;
}

Eigen::Matrix4f MakePerspective(float fovYDegrees, float aspect, float zNear,
                                float zFar) {
  const float fovYRadians = fovYDegrees * 3.14159265358979323846f / 180.0f;
  const float tanHalfFov = std::tan(fovYRadians * 0.5f);

  Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
  proj(0, 0) = 1.0f / (aspect * tanHalfFov);
  proj(1, 1) = 1.0f / tanHalfFov;
  proj(2, 2) = -(zFar + zNear) / (zFar - zNear);
  proj(2, 3) = -(2.0f * zFar * zNear) / (zFar - zNear);
  proj(3, 2) = -1.0f;

  return proj;
}