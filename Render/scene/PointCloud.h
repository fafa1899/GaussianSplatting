#pragma once

#include <Eigen/Core>
#include <vector>

struct Point3D {
  Eigen::Vector3f position = Eigen::Vector3f::Zero();
  Eigen::Vector3f color = Eigen::Vector3f::Ones();
  float opacity = 1.0f;
  Eigen::Vector3f scale = Eigen::Vector3f::Ones();
  Eigen::Vector4f rotation = Eigen::Vector4f(1.0f, 0.0f, 0.0f, 0.0f);
};

class PointCloud {
 public:
  std::vector<Point3D> points;
};