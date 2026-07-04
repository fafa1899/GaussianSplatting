#pragma once

#include <Eigen/Core>

Eigen::Matrix4f MakeLookAt(const Eigen::Vector3f& eye,
                           const Eigen::Vector3f& target,
                           const Eigen::Vector3f& up);

Eigen::Matrix4f MakePerspective(float fovYDegrees, float aspect, float zNear,
                                float zFar);