#include "scene/Camera.h"

#include "math/Transform.h"

Camera Camera::CreateLookAt(int width, int height, float fovYDegrees,
                            float zNear, float zFar, const Eigen::Vector3f& eye,
                            const Eigen::Vector3f& target,
                            const Eigen::Vector3f& up) {
  Camera camera;
  camera.width = width;
  camera.height = height;
  camera.fovYDegrees = fovYDegrees;
  camera.zNear = zNear;
  camera.zFar = zFar;
  camera.view = MakeLookAt(eye, target, up);
  camera.proj = MakePerspective(
      fovYDegrees, static_cast<float>(width) / static_cast<float>(height),
      zNear, zFar);
  return camera;
}