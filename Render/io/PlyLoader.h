#pragma once

#include <string>

#include "scene/PointCloud.h"

class PlyLoader {
 public:
  static PointCloud LoadPointCloud(const std::string& path);
};