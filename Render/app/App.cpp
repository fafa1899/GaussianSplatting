#include "app/App.h"

#include <iostream>
#include <string>

#include "io/PlyLoader.h"
#include "render/PointRenderer.h"
#include "scene/Camera.h"

int App::Run() const {
  const std::string plyPath =
      R"(D:\3DGS\Unity3DGS\models\bicycle\point_cloud\iteration_30000\point_cloud.ply)";
  const std::string outputPath =
      R"(D:\3DGS\Unity3DGS\GaussianSplatting\Render\output\bicycle_points_opacity.png)";

  std::cout << "Loading PLY: " << plyPath << std::endl;
  PointCloud cloud = PlyLoader::LoadPointCloud(plyPath);
  std::cout << "Point count: " << cloud.points.size() << std::endl;

  Camera camera = Camera::CreateLookAt(
      1280, 720, 60.0f, 0.1f, 1000.0f, Eigen::Vector3f(0.0f, 0.0f, 0.0f),
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