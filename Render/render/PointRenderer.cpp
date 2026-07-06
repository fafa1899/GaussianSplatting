#include "render/PointRenderer.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <limits>
#include <vector>

static constexpr float SH_C0 = 0.28209479177387814f;
static constexpr float SH_C1 = 0.4886025119029199f;

static constexpr float SH_C2[5] = {1.0925484305920792f, -1.0925484305920792f,
                                   0.31539156525252005f, -1.0925484305920792f,
                                   0.5462742152960396f};

static constexpr float SH_C3[7] = {-0.5900435899266435f, 2.890611442640554f,
                                   -0.4570457994644658f, 0.3731763325901154f,
                                   -0.4570457994644658f, 1.445305721320277f,
                                   -0.5900435899266435f};

static constexpr int kTileWidth = 16;
static constexpr int kTileHeight = 16;

struct TileEntry {
  int tileId = 0;
  int gaussianIndex = 0;
  float depth = 0.0f;
};

struct TileRange {
  int begin = 0;
  int end = 0;
  bool valid = false;
};

// 投影到屏幕后的3D点
struct ProjectedGaussian {
  // 椭圆中心的屏幕坐标
  int px = 0;
  int py = 0;

  // 深度，用于排序，决定先画远处还是近处
  float depth = 0.0f;

  // 遍历像素时用的包围盒半径
  int radiusX = 1;
  int radiusY = 1;

  // 2D 椭圆核最关键的参数
  // quad = A*dx^2 + 2B*dx*dy + C*dy^2
  // 椭圆度量:某个像素相对椭圆中心的“距离”
  float conicA = 1.0f;
  float conicB = 0.0f;
  float conicC = 1.0f;

  Eigen::Vector3f color = Eigen::Vector3f::Ones();
  float opacity = 1.0f;
};

struct PreprocessStats {
  size_t totalInput = 0;
  size_t clippedBehindCamera = 0;
  size_t clippedOutsideNdc = 0;
  size_t clippedDegenerateCov = 0;
  size_t clippedBadEigen = 0;
  size_t projectedCount = 0;

  int minBBox = std::numeric_limits<int>::max();
  int maxBBox = 0;
  double sumBBox = 0.0;

  size_t bboxGe16 = 0;
  size_t bboxGe32 = 0;
  size_t bboxGe64 = 0;

  float minOpacity = std::numeric_limits<float>::max();
  float maxOpacity = 0.0f;
  double sumOpacity = 0.0;
};

static float EvalSHChannel(const float* sh, const Eigen::Vector3f& dir) {
  const float x = dir.x();
  const float y = dir.y();
  const float z = dir.z();

  float result = SH_C0 * sh[0];

  result += -SH_C1 * y * sh[1];
  result += SH_C1 * z * sh[2];
  result += -SH_C1 * x * sh[3];

  const float xx = x * x;
  const float yy = y * y;
  const float zz = z * z;
  const float xy = x * y;
  const float yz = y * z;
  const float xz = x * z;

  result += SH_C2[0] * xy * sh[4];
  result += SH_C2[1] * yz * sh[5];
  result += SH_C2[2] * (2.0f * zz - xx - yy) * sh[6];
  result += SH_C2[3] * xz * sh[7];
  result += SH_C2[4] * (xx - yy) * sh[8];

  result += SH_C3[0] * y * (3.0f * xx - yy) * sh[9];
  result += SH_C3[1] * xy * z * sh[10];
  result += SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11];
  result += SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12];
  result += SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13];
  result += SH_C3[5] * z * (xx - yy) * sh[14];
  result += SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];

  return result;
}

// 四元数转旋转矩阵
static Eigen::Matrix3f QuaternionToMatrix(const Eigen::Vector4f& qInput) {
  Eigen::Vector4f q = qInput.normalized();

  const float w = q.x();
  const float x = q.y();
  const float y = q.z();
  const float z = q.w();

  Eigen::Matrix3f R;
  R(0, 0) = 1.0f - 2.0f * (y * y + z * z);
  R(0, 1) = 2.0f * (x * y - w * z);
  R(0, 2) = 2.0f * (x * z + w * y);

  R(1, 0) = 2.0f * (x * y + w * z);
  R(1, 1) = 1.0f - 2.0f * (x * x + z * z);
  R(1, 2) = 2.0f * (y * z - w * x);

  R(2, 0) = 2.0f * (x * z - w * y);
  R(2, 1) = 2.0f * (y * z + w * x);
  R(2, 2) = 1.0f - 2.0f * (x * x + y * y);
  return R;
}

static Eigen::Vector3f EvalSHColor(const std::array<float, 48>& sh,
                                   const Eigen::Vector3f& dir) {
  Eigen::Vector3f color;
  color.x() = EvalSHChannel(&sh[0], dir);
  color.y() = EvalSHChannel(&sh[16], dir);
  color.z() = EvalSHChannel(&sh[32], dir);

  // 对齐官方：+0.5 再 clamp
  color.array() += 0.5f;
  color = color.cwiseMax(0.0f);
  color = color.cwiseMin(1.0f);
  return color;
}

// 计算3D协方差矩阵，Gaussian 在 3D 空间里的“形状描述”
static Eigen::Matrix3f BuildWorldCovariance(const Eigen::Vector3f& scale,
                                            const Eigen::Vector4f& rotation) {
  Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
  S(0, 0) = scale.x();
  S(1, 1) = scale.y();
  S(2, 2) = scale.z();

  const Eigen::Matrix3f R = QuaternionToMatrix(rotation);
  const Eigen::Matrix3f M = S * R;
  return M.transpose() * M;
}

static std::vector<ProjectedGaussian> PreprocessGaussians(
    const PointCloud& cloud, const Camera& camera, PreprocessStats& stats) {
  stats.totalInput = cloud.points.size();

  // 视图投影矩阵
  const Eigen::Matrix4f viewProj = camera.proj * camera.view;

  // FOV转成“像素焦距”
  const float fovYRadians =
      camera.fovYDegrees * 3.14159265358979323846f / 180.0f;
  const float focalPixels =
      static_cast<float>(camera.height) / (2.0f * std::tan(fovYRadians * 0.5f));

  // SH 需要世界空间里，从相机指向点的方向
  const Eigen::Matrix4f invView = camera.view.inverse();
  const Eigen::Vector3f cameraCenter = invView.block<3, 1>(0, 3);

  std::vector<ProjectedGaussian> projected;
  projected.reserve(cloud.points.size());

  int debugPrinted = 0;

  // 逐个 3D Gaussian 做预处理
  for (const Point3D& point : cloud.points) {
    // 投影到裁剪空间
    Eigen::Vector4f world(point.position.x(), point.position.y(),
                          point.position.z(), 1.0f);
    Eigen::Vector4f clip = viewProj * world;
    if (clip.w() <= 0.0f) {
      ++stats.clippedBehindCamera;
      continue;
    }

    // 归一化设备坐标 NDC
    Eigen::Vector3f ndc = clip.head<3>() / clip.w();
    if (ndc.x() < -1.0f || ndc.x() > 1.0f || ndc.y() < -1.0f ||
        ndc.y() > 1.0f || ndc.z() < -1.0f || ndc.z() > 1.0f) {
      ++stats.clippedOutsideNdc;
      continue;
    }

    // NDC 到像素坐标
    const int px = static_cast<int>((ndc.x() * 0.5f + 0.5f) *
                                    static_cast<float>(camera.width - 1));
    const int py = static_cast<int>((ndc.y() * 0.5f + 0.5f) *
                                    static_cast<float>(camera.height - 1));
    if (px < 0 || px >= camera.width || py < 0 || py >= camera.height) {
      continue;
    }

    // 世界高斯转到相机空间
    const Eigen::Vector4f viewPos4 = camera.view * world;
    Eigen::Vector3f viewPos = viewPos4.head<3>();

    // 世界协方差
    const Eigen::Matrix3f covWorld =
        BuildWorldCovariance(point.scale, point.rotation);

    // 相机协方差
    const Eigen::Matrix3f viewRot = camera.view.block<3, 3>(0, 0);
    const Eigen::Matrix3f covView = viewRot * covWorld * viewRot.transpose();

    // 对齐官方 computeCov2D：
    // 先在相机空间限制 x/z 和 y/z 的斜率，避免极端透视把椭圆拉爆。
    const float tanFovY = std::tan(fovYRadians * 0.5f);
    const float tanFovX = tanFovY * static_cast<float>(camera.width) /
                          static_cast<float>(camera.height);

    const float limx = 1.3f * tanFovX;
    const float limy = 1.3f * tanFovY;

    const float safeZ = std::max(std::abs(viewPos.z()), 1e-4f);
    const float txtz = viewPos.x() / safeZ;
    const float tytz = viewPos.y() / safeZ;

    viewPos.x() = std::min(limx, std::max(-limx, txtz)) * safeZ;
    viewPos.y() = std::min(limy, std::max(-limy, tytz)) * safeZ;

    // 投影 Jacobian
    const float z = std::max(std::abs(viewPos.z()), 1e-4f);
    const float fx = focalPixels;
    const float fy = focalPixels;

    Eigen::Matrix<float, 2, 3> J;
    J(0, 0) = fx / z;
    J(0, 1) = 0.0f;
    J(0, 2) = -fx * viewPos.x() / (z * z);

    J(1, 0) = 0.0f;
    J(1, 1) = fy / z;
    J(1, 2) = -fy * viewPos.y() / (z * z);

    // 3D 协方差投影成 2D 协方差
    Eigen::Matrix2f cov2D = J * covView * J.transpose();

    // 对齐官方思路：先记录加稳定项前的 determinant
    const float detBefore = cov2D.determinant();

    // 数值稳定项，避免协方差过小或退化
    cov2D(0, 0) += 0.3f;
    cov2D(1, 1) += 0.3f;

    const float detAfter =
        cov2D(0, 0) * cov2D(1, 1) - cov2D(0, 1) * cov2D(1, 0);
    if (detAfter <= 1e-8f) {
      ++stats.clippedDegenerateCov;
      continue;
    }

    // 对齐官方 antialiasing / h_convolution_scaling 的思路：
    // 稳定项把核摊大之后，适当缩小 opacity，避免能量过度扩散
    float opacityScale = 1.0f;
    if (detBefore > 1e-8f) {
      opacityScale = std::sqrt(std::max(0.000025f, detBefore / detAfter));
    }

    // 协方差逆矩阵：得到 conic
    const Eigen::Matrix2f invCov2D = cov2D.inverse();

    const float conicA = invCov2D(0, 0);
    const float conicB = invCov2D(0, 1);
    const float conicC = invCov2D(1, 1);

    // 对齐官方 forward.cu：
    // cov2D = [a b; b c]
    const float a = cov2D(0, 0);
    const float b = cov2D(0, 1);
    const float c = cov2D(1, 1);

    const float mid = 0.5f * (a + c);
    const float discr = std::max(0.1f, mid * mid - detAfter);

    const float lambda1 = mid + std::sqrt(discr);
    const float lambda2 = mid - std::sqrt(discr);
    const float lambdaMax = std::max(lambda1, lambda2);

    // 仍然保留 bbox 上限，避免个别高斯失控
    const int bboxRadius = std::clamp(
        static_cast<int>(std::ceil(3.0f * std::sqrt(lambdaMax))), 1, 64);

    // 计算 SH 颜色
    Eigen::Vector3f viewDir = point.position - cameraCenter;
    if (viewDir.norm() < 1e-8f) {
      continue;
    }
    viewDir.normalize();
    const Eigen::Vector3f shColor = EvalSHColor(point.sh, viewDir);

    //
    // const float finalOpacity =
    //    std::clamp(point.opacity * opacityScale, 0.0f, 1.0f);
    const float opacityScaleBlend = 0.5f;
    const float blendedOpacityScale =
        1.0f + (opacityScale - 1.0f) * opacityScaleBlend;
    const float finalOpacity =
        std::clamp(point.opacity * blendedOpacityScale, 0.0f, 1.0f);

    //
    ++stats.projectedCount;

    stats.minBBox = std::min(stats.minBBox, bboxRadius);
    stats.maxBBox = std::max(stats.maxBBox, bboxRadius);
    stats.sumBBox += static_cast<double>(bboxRadius);

    if (bboxRadius >= 16) ++stats.bboxGe16;
    if (bboxRadius >= 32) ++stats.bboxGe32;
    if (bboxRadius >= 64) ++stats.bboxGe64;

    stats.minOpacity = std::min(stats.minOpacity, finalOpacity);
    stats.maxOpacity = std::max(stats.maxOpacity, finalOpacity);
    stats.sumOpacity += static_cast<double>(finalOpacity);

    // 收集成 projected 列表
    ProjectedGaussian p;
    p.px = px;
    p.py = py;
    p.depth = std::abs(viewPos.z());
    p.radiusX = bboxRadius;
    p.radiusY = bboxRadius;
    p.conicA = conicA;
    p.conicB = conicB;
    p.conicC = conicC;
    p.color = shColor;
    p.opacity = finalOpacity;
    projected.push_back(p);
  }

  return projected;
}

// 深度排序
static void SortGaussiansByDepth(std::vector<ProjectedGaussian>& gaussians) {
  std::sort(gaussians.begin(), gaussians.end(),
            [](const ProjectedGaussian& a, const ProjectedGaussian& b) {
              return a.depth > b.depth;
            });
}

static cv::Mat RasterizeGaussians(
    const std::vector<ProjectedGaussian>& gaussians, const Camera& camera) {
  cv::Mat image(camera.height, camera.width, CV_32FC3,
                cv::Scalar(20.0f / 255.0f, 20.0f / 255.0f, 20.0f / 255.0f));

  // 逐个椭圆 splat 到图像
  for (const ProjectedGaussian& point : gaussians) {
    // 先取 bbox 范围
    const int minX = std::max(0, point.px - point.radiusX);
    const int maxX = std::min(camera.width - 1, point.px + point.radiusX);
    const int minY = std::max(0, point.py - point.radiusY);
    const int maxY = std::min(camera.height - 1, point.py + point.radiusY);

    // 基础 alpha 和颜色
    const float alpha = std::clamp(point.opacity, 0.0f, 1.0f);
    const cv::Vec3f srcColor(point.color.z(), point.color.y(), point.color.x());

    for (int y = minY; y <= maxY; ++y) {
      for (int x = minX; x <= maxX; ++x) {
        const float dx = static_cast<float>(x - point.px);
        const float dy = static_cast<float>(y - point.py);

        // 椭圆二次型
        const float quad = point.conicA * dx * dx +
                           2.0f * point.conicB * dx * dy +
                           point.conicC * dy * dy;

        // 椭圆高斯权重
        const float power = -0.5f * quad;
        if (power < -12.0f) {
          continue;
        }

        const float weight = std::exp(power);
        if (weight < 1e-4f) {
          continue;
        }

        // 局部 alpha
        const float localAlpha = alpha * weight;
        if (localAlpha <= 1e-4f) {
          continue;
        }

        // 最终颜色混合
        cv::Vec3f& dst = image.at<cv::Vec3f>(y, x);
        dst = srcColor * localAlpha + dst * (1.0f - localAlpha);
      }
    }
  }

  // 转回 8 位图
  cv::Mat output;
  image.convertTo(output, CV_8UC3, 255.0);
  return output;
}

static std::vector<TileEntry> BuildTileEntries(
    const std::vector<ProjectedGaussian>& gaussians, const Camera& camera) {
  const int tileCountX = (camera.width + kTileWidth - 1) / kTileWidth;
  const int tileCountY = (camera.height + kTileHeight - 1) / kTileHeight;

  std::vector<TileEntry> entries;
  entries.reserve(gaussians.size() * 4);  // 先给个保守初值，后面可以调

  for (int i = 0; i < static_cast<int>(gaussians.size()); ++i) {
    const ProjectedGaussian& g = gaussians[i];

    const int minX = std::max(0, g.px - g.radiusX);
    const int maxX = std::min(camera.width - 1, g.px + g.radiusX);
    const int minY = std::max(0, g.py - g.radiusY);
    const int maxY = std::min(camera.height - 1, g.py + g.radiusY);

    const int tileMinX = std::max(0, minX / kTileWidth);
    const int tileMaxX = std::min(tileCountX - 1, maxX / kTileWidth);
    const int tileMinY = std::max(0, minY / kTileHeight);
    const int tileMaxY = std::min(tileCountY - 1, maxY / kTileHeight);

    for (int ty = tileMinY; ty <= tileMaxY; ++ty) {
      for (int tx = tileMinX; tx <= tileMaxX; ++tx) {
        TileEntry e;
        e.tileId = ty * tileCountX + tx;
        e.gaussianIndex = i;
        e.depth = g.depth;
        entries.push_back(e);
      }
    }
  }

  return entries;
}

static std::vector<TileRange> BuildTileRanges(std::vector<TileEntry>& entries,
                                              const Camera& camera) {
  const int tileCountX = (camera.width + kTileWidth - 1) / kTileWidth;
  const int tileCountY = (camera.height + kTileHeight - 1) / kTileHeight;
  const int tileCount = tileCountX * tileCountY;

  std::sort(entries.begin(), entries.end(),
            [](const TileEntry& a, const TileEntry& b) {
              if (a.tileId != b.tileId) {
                return a.tileId < b.tileId;
              }
              // return a.depth > b.depth;  // 远到近
              return a.depth < b.depth;  // 近到远
            });

  std::vector<TileRange> ranges(tileCount);

  if (entries.empty()) {
    return ranges;
  }

  int currentTile = entries[0].tileId;
  ranges[currentTile].begin = 0;
  ranges[currentTile].valid = true;

  for (int i = 1; i < static_cast<int>(entries.size()); ++i) {
    const int prevTile = entries[i - 1].tileId;
    const int currTile = entries[i].tileId;

    if (currTile != prevTile) {
      ranges[prevTile].end = i;
      ranges[currTile].begin = i;
      ranges[currTile].valid = true;
    }
  }

  ranges[entries.back().tileId].end = static_cast<int>(entries.size());
  return ranges;
}

static cv::Mat RasterizeTiles(const std::vector<ProjectedGaussian>& gaussians,
                              const std::vector<TileEntry>& entries,
                              const std::vector<TileRange>& ranges,
                              const Camera& camera) {
  const cv::Vec3f backgroundColor(20.0f / 255.0f, 20.0f / 255.0f,
                                  20.0f / 255.0f);

  cv::Mat finalImage(
      camera.height, camera.width, CV_32FC3,
      cv::Scalar(backgroundColor[0], backgroundColor[1], backgroundColor[2]));

  cv::Mat transmittanceDebug(camera.height, camera.width, CV_32FC1,
                             cv::Scalar(1.0f));

  cv::Mat contribCountDebug(camera.height, camera.width, CV_32SC1,
                            cv::Scalar(0));

  const int tileCountX = (camera.width + kTileWidth - 1) / kTileWidth;
  const int tileCountY = (camera.height + kTileHeight - 1) / kTileHeight;

  for (int ty = 0; ty < tileCountY; ++ty) {
    for (int tx = 0; tx < tileCountX; ++tx) {
      const int tileId = ty * tileCountX + tx;
      const TileRange& range = ranges[tileId];
      if (!range.valid) {
        continue;
      }

      const int tileMinX = tx * kTileWidth;
      const int tileMaxX = std::min(camera.width, tileMinX + kTileWidth);
      const int tileMinY = ty * kTileHeight;
      const int tileMaxY = std::min(camera.height, tileMinY + kTileHeight);

      // 改成：tile -> pixel -> gaussian
      for (int y = tileMinY; y < tileMaxY; ++y) {
        for (int x = tileMinX; x < tileMaxX; ++x) {
          float T = 1.0f;
          cv::Vec3f C(0.0f, 0.0f, 0.0f);

          int contribCount = 0;

          for (int ei = range.begin; ei < range.end; ++ei) {
            const ProjectedGaussian& point =
                gaussians[entries[ei].gaussianIndex];

            // 先做一个 bbox 粗筛，避免不必要计算
            if (x < point.px - point.radiusX || x > point.px + point.radiusX ||
                y < point.py - point.radiusY || y > point.py + point.radiusY) {
              continue;
            }

            const float dx = static_cast<float>(x - point.px);
            const float dy = static_cast<float>(y - point.py);

            const float quad = point.conicA * dx * dx +
                               2.0f * point.conicB * dx * dy +
                               point.conicC * dy * dy;

            const float power = -0.5f * quad;

            // 对齐官方：power > 0 直接跳过
            if (power > 0.0f) {
              continue;
            }

            const float alpha =
                std::min(0.99f, point.opacity * std::exp(power));

            if (alpha < (1.0f / 255.0f)) {
              continue;
            }

            const float testT = T * (1.0f - alpha);

            C += cv::Vec3f(point.color.z(), point.color.y(), point.color.x()) *
                 (alpha * T);

            T = testT;

            ++contribCount;

            // 这里终于和官方语义更接近：
            // 当前像素已经足够不透明，就停止处理后续 Gaussian
            if (T < 1e-4f) {
              break;
            }
          }

          finalImage.at<cv::Vec3f>(y, x) = C + backgroundColor * T;
          transmittanceDebug.at<float>(y, x) = T;
          contribCountDebug.at<int>(y, x) = contribCount;
        }
      }
    }
  }

  cv::Mat output;
  finalImage.convertTo(output, CV_8UC3, 255.0);

  // Debug 1: transmittance，T 越小越黑，说明越早被吃光
  cv::Mat transmittance8U;
  transmittanceDebug.convertTo(transmittance8U, CV_8UC1, 255.0);
  cv::imwrite(
      R"(D:\3DGS\Unity3DGS\GaussianSplatting\Render\output\transmittance_debug.png)",
      transmittance8U);

  // Debug 2: contrib count，先归一化到 0~255 方便看
  double minCount = 0.0;
  double maxCount = 0.0;
  cv::minMaxLoc(contribCountDebug, &minCount, &maxCount);

  cv::Mat contribCount8U;
  if (maxCount > 0.0) {
    contribCountDebug.convertTo(contribCount8U, CV_8UC1, 255.0 / maxCount);
  } else {
    contribCount8U =
        cv::Mat(camera.height, camera.width, CV_8UC1, cv::Scalar(0));
  }

  cv::imwrite(
      R"(D:\3DGS\Unity3DGS\GaussianSplatting\Render\output\contrib_count_debug.png)",
      contribCount8U);

  std::cout << "Contrib count min/max: " << minCount << " / " << maxCount
            << std::endl;

  return output;
}

cv::Mat PointRenderer::Render(const PointCloud& cloud,
                              const Camera& camera) const {
  PreprocessStats stats;

  std::vector<ProjectedGaussian> projected =
      PreprocessGaussians(cloud, camera, stats);

  std::cout << "=== Preprocess Stats ===" << std::endl;
  std::cout << "Input: " << stats.totalInput << std::endl;
  std::cout << "Projected: " << stats.projectedCount << std::endl;
  std::cout << "Behind camera: " << stats.clippedBehindCamera << std::endl;
  std::cout << "Outside NDC: " << stats.clippedOutsideNdc << std::endl;
  std::cout << "Degenerate cov: " << stats.clippedDegenerateCov << std::endl;
  std::cout << "Bad eigen: " << stats.clippedBadEigen << std::endl;

  if (stats.projectedCount > 0) {
    std::cout << "BBox min/max/avg: " << stats.minBBox << " / " << stats.maxBBox
              << " / "
              << (stats.sumBBox / static_cast<double>(stats.projectedCount))
              << std::endl;

    std::cout << "BBox >=16: " << stats.bboxGe16 << std::endl;
    std::cout << "BBox >=32: " << stats.bboxGe32 << std::endl;
    std::cout << "BBox >=64: " << stats.bboxGe64 << std::endl;

    std::cout << "Opacity min/max/avg: " << stats.minOpacity << " / "
              << stats.maxOpacity << " / "
              << (stats.sumOpacity / static_cast<double>(stats.projectedCount))
              << std::endl;
  }

  // SortGaussiansByDepth(projected);

  // return RasterizeGaussians(projected, camera);

  std::vector<TileEntry> entries = BuildTileEntries(projected, camera);
  std::vector<TileRange> ranges = BuildTileRanges(entries, camera);

  std::cout << "Tile entries: " << entries.size() << std::endl;

  return RasterizeTiles(projected, entries, ranges, camera);
}