#include "app/App.h"

#include <iostream>
#include <string>

#include "io/PlyLoader.h"
#include "render/PointRenderer.h"
#include "scene/Camera.h"

static void ComputeBounds(const PointCloud& cloud, Eigen::Vector3f& outMin,
                          Eigen::Vector3f& outMax) {
  outMin = Eigen::Vector3f(std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max());

  outMax = Eigen::Vector3f(std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest());

  for (const Point3D& point : cloud.points) {
    outMin = outMin.cwiseMin(point.position);
    outMax = outMax.cwiseMax(point.position);
  }
}

int App::Run() const {
  const std::string plyPath =
      R"(D:\3DGS\Unity3DGS\models\bicycle\point_cloud\iteration_30000\point_cloud.ply)";
  const std::string outputPath =
      R"(D:\3DGS\Unity3DGS\GaussianSplatting\Render\output\bicycle_points_opacity.png)";

  std::cout << "Loading PLY: " << plyPath << std::endl;
  PointCloud cloud = PlyLoader::LoadPointCloud(plyPath);
  std::cout << "Point count: " << cloud.points.size() << std::endl;

  Eigen::Vector3f boundsMin;
  Eigen::Vector3f boundsMax;
  ComputeBounds(cloud, boundsMin, boundsMax);

  const Eigen::Vector3f center = 0.5f * (boundsMin + boundsMax);
  const Eigen::Vector3f extent = boundsMax - boundsMin;
  const float radius = 0.5f * extent.norm();

  std::cout << "Bounds min: " << boundsMin.transpose() << std::endl;
  std::cout << "Bounds max: " << boundsMax.transpose() << std::endl;
  std::cout << "Center: " << center.transpose() << std::endl;
  std::cout << "Radius: " << radius << std::endl;

  const Eigen::Vector3f forward = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  const Eigen::Vector3f up = Eigen::Vector3f(0.0f, 1.0f, 0.0f);

  // 稍微从右上后方看向中心，先做一个通用调试视角
  Eigen::Vector3f viewDir = Eigen::Vector3f(-0.35f, 0.15f, -1.0f).normalized();

  // 按包围球半径退后，2.5f 这个系数是经验值，足够先把主体放进画面
  const float cameraDistance = std::max(radius * 2.5f, 1.0f);

  // 让相机看向整体中心，稍微抬一点点，避免只盯着地面
  const Eigen::Vector3f target =
      center + Eigen::Vector3f(0.0f, extent.y() * 0.05f, 0.0f);
  const Eigen::Vector3f eye = target - viewDir * cameraDistance;

  std::cout << "Eye: " << eye.transpose() << std::endl;
  std::cout << "Target: " << target.transpose() << std::endl;

  // Camera camera = Camera::CreateLookAt(1280, 720, 60.0f, 0.1f,
  //                                     std::max(1000.0f, cameraDistance
  //                                     * 4.0f), eye, target, up);

  Camera camera = Camera::CreateLookAt(
      1280, 720, 60.0f, 0.1f, 1000.0f, Eigen::Vector3f(-1.0f, 0.0f, -1.0f),
      Eigen::Vector3f(0.0f, 0.0f, 1.0f), Eigen::Vector3f(0.0f, 1.0f, 0.0f));

  PointRenderer renderer;
  cv::Mat image = renderer.Render(cloud, camera);

  if (!cv::imwrite(outputPath, image)) {
    std::cerr << "Failed to save image: " << outputPath << std::endl;
    return 1;
  }

  std::cout << "Saved image to: " << outputPath << std::endl;
  return 0;

  return 0;
}