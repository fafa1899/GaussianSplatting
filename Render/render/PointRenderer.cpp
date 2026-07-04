#include "render/PointRenderer.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <limits>
#include <vector>

struct ProjectedPoint {
  int px = 0;
  int py = 0;
  float depth = 0.0f;

  int radiusX = 1;
  int radiusY = 1;

  float conicA = 1.0f;
  float conicB = 0.0f;
  float conicC = 1.0f;

  Eigen::Vector3f color = Eigen::Vector3f::Ones();
  float opacity = 1.0f;
};

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

cv::Mat PointRenderer::Render(const PointCloud& cloud,
                              const Camera& camera) const {
  cv::Mat image(camera.height, camera.width, CV_32FC3,
                cv::Scalar(20.0f / 255.0f, 20.0f / 255.0f, 20.0f / 255.0f));

  const Eigen::Matrix4f viewProj = camera.proj * camera.view;

  const float fovYRadians =
      camera.fovYDegrees * 3.14159265358979323846f / 180.0f;
  const float focalPixels =
      static_cast<float>(camera.height) / (2.0f * std::tan(fovYRadians * 0.5f));

  std::vector<ProjectedPoint> projected;
  projected.reserve(cloud.points.size());

  for (const Point3D& point : cloud.points) {
    Eigen::Vector4f world(point.position.x(), point.position.y(),
                          point.position.z(), 1.0f);
    Eigen::Vector4f clip = viewProj * world;

    if (clip.w() <= 0.0f) {
      continue;
    }

    Eigen::Vector3f ndc = clip.head<3>() / clip.w();

    if (ndc.x() < -1.0f || ndc.x() > 1.0f || ndc.y() < -1.0f ||
        ndc.y() > 1.0f || ndc.z() < -1.0f || ndc.z() > 1.0f) {
      continue;
    }

    const int px = static_cast<int>((ndc.x() * 0.5f + 0.5f) *
                                    static_cast<float>(camera.width - 1));
    const int py = static_cast<int>((ndc.y() * 0.5f + 0.5f) *
                                    static_cast<float>(camera.height - 1));

    if (px < 0 || px >= camera.width || py < 0 || py >= camera.height) {
      continue;
    }

    const Eigen::Vector4f viewPos4 = camera.view * world;
    const Eigen::Vector3f viewPos = viewPos4.head<3>();

    const Eigen::Matrix3f covWorld =
        BuildWorldCovariance(point.scale, point.rotation);

    const Eigen::Matrix3f viewRot = camera.view.block<3, 3>(0, 0);
    const Eigen::Matrix3f covView = viewRot * covWorld * viewRot.transpose();

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

    Eigen::Matrix2f cov2D = J * covView * J.transpose();

    // 数值稳定项，避免协方差过小或退化
    cov2D(0, 0) += 0.3f;
    cov2D(1, 1) += 0.3f;

    const float det = cov2D.determinant();
    if (det <= 1e-8f) {
      continue;
    }

    const Eigen::Matrix2f invCov2D = cov2D.inverse();

    const float conicA = invCov2D(0, 0);
    const float conicB = invCov2D(0, 1);
    const float conicC = invCov2D(1, 1);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(cov2D);
    if (solver.info() != Eigen::Success) {
      continue;
    }

    const Eigen::Vector2f eigenValues = solver.eigenvalues();
    const float lambdaMax = std::max(eigenValues.x(), eigenValues.y());

    // 先用统一 bbox 半径，保证包得住椭圆
    const int bboxRadius = std::clamp(
        static_cast<int>(std::ceil(3.0f * std::sqrt(lambdaMax))), 1, 64);

    ProjectedPoint p;
    p.px = px;
    p.py = py;
    p.depth = ndc.z();
    p.radiusX = bboxRadius;
    p.radiusY = bboxRadius;
    p.conicA = conicA;
    p.conicB = conicB;
    p.conicC = conicC;
    p.color = point.color;
    p.opacity = point.opacity;
    projected.push_back(p);
  }

  std::sort(projected.begin(), projected.end(),
            [](const ProjectedPoint& a, const ProjectedPoint& b) {
              return a.depth > b.depth;
            });

  for (const ProjectedPoint& point : projected) {
    const int minX = std::max(0, point.px - point.radiusX);
    const int maxX = std::min(camera.width - 1, point.px + point.radiusX);
    const int minY = std::max(0, point.py - point.radiusY);
    const int maxY = std::min(camera.height - 1, point.py + point.radiusY);

    const float alpha = std::clamp(point.opacity, 0.0f, 1.0f);
    const cv::Vec3f srcColor(point.color.z(), point.color.y(), point.color.x());

    for (int y = minY; y <= maxY; ++y) {
      for (int x = minX; x <= maxX; ++x) {
        const float dx = static_cast<float>(x - point.px);
        const float dy = static_cast<float>(y - point.py);

        const float quad = point.conicA * dx * dx +
                           2.0f * point.conicB * dx * dy +
                           point.conicC * dy * dy;

        const float power = -0.5f * quad;
        if (power < -12.0f) {
          continue;
        }

        const float weight = std::exp(power);
        if (weight < 1e-4f) {
          continue;
        }

        const float localAlpha = alpha * weight;
        if (localAlpha <= 1e-4f) {
          continue;
        }

        cv::Vec3f& dst = image.at<cv::Vec3f>(y, x);
        dst = srcColor * localAlpha + dst * (1.0f - localAlpha);
      }
    }
  }

  cv::Mat output;
  image.convertTo(output, CV_8UC3, 255.0);
  return output;
}