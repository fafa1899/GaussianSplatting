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
    const Eigen::Vector3f viewPos = viewPos4.head<3>();

    // 世界协方差
    const Eigen::Matrix3f covWorld =
        BuildWorldCovariance(point.scale, point.rotation);

    // 相机协方差
    const Eigen::Matrix3f viewRot = camera.view.block<3, 3>(0, 0);
    const Eigen::Matrix3f covView = viewRot * covWorld * viewRot.transpose();

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

    // 数值稳定项，避免协方差过小或退化
    cov2D(0, 0) += 0.3f;
    cov2D(1, 1) += 0.3f;

    const float det = cov2D.determinant();
    if (det <= 1e-8f) {
      ++stats.clippedDegenerateCov;
      continue;
    }

    // 协方差逆矩阵：得到 conic
    const Eigen::Matrix2f invCov2D = cov2D.inverse();

    const float conicA = invCov2D(0, 0);
    const float conicB = invCov2D(0, 1);
    const float conicC = invCov2D(1, 1);

    // 用特征值估一个 bbox
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(cov2D);
    if (solver.info() != Eigen::Success) {
      ++stats.clippedBadEigen;
      continue;
    }

    const Eigen::Vector2f eigenValues = solver.eigenvalues();
    const float lambdaMax = std::max(eigenValues.x(), eigenValues.y());

    // 先用统一 bbox 半径，保证包得住椭圆
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
    ++stats.projectedCount;

    stats.minBBox = std::min(stats.minBBox, bboxRadius);
    stats.maxBBox = std::max(stats.maxBBox, bboxRadius);
    stats.sumBBox += static_cast<double>(bboxRadius);

    if (bboxRadius >= 16) ++stats.bboxGe16;
    if (bboxRadius >= 32) ++stats.bboxGe32;
    if (bboxRadius >= 64) ++stats.bboxGe64;

    stats.minOpacity = std::min(stats.minOpacity, point.opacity);
    stats.maxOpacity = std::max(stats.maxOpacity, point.opacity);
    stats.sumOpacity += static_cast<double>(point.opacity);

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
    p.opacity = point.opacity;
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

  SortGaussiansByDepth(projected);

  return RasterizeGaussians(projected, camera);
}