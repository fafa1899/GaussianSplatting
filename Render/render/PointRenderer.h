#pragma once

#include <opencv2/opencv.hpp>

#include "scene/Camera.h"
#include "scene/PointCloud.h"

class PointRenderer {
 public:
  cv::Mat Render(const PointCloud& cloud, const Camera& camera) const;
};