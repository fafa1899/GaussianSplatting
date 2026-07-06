#include "PlyLoader.h"

#include <algorithm>
#include <fstream>
#include <iostream>

#include "PlyGaussianVertex.h"

using namespace std;

struct PropertyInfo {
  std::string type;
  std::string name;
};

static constexpr float kShC0 = 0.28209479177387814f;

static Eigen::Vector3f DecodeColorFromDC(const float dc[3]) {
  Eigen::Vector3f color;
  color.x() = std::clamp(dc[0] * kShC0 + 0.5f, 0.0f, 1.0f);
  color.y() = std::clamp(dc[1] * kShC0 + 0.5f, 0.0f, 1.0f);
  color.z() = std::clamp(dc[2] * kShC0 + 0.5f, 0.0f, 1.0f);
  return color;
}

static size_t GetPropertyByteLength(const std::string& type) {
  if (type == "float" || type == "float32") return sizeof(float);
  if (type == "double" || type == "float64") return sizeof(double);
  if (type == "uchar" || type == "uint8") return sizeof(std::uint8_t);
  if (type == "char" || type == "int8") return sizeof(std::int8_t);
  if (type == "ushort" || type == "uint16") return sizeof(std::uint16_t);
  if (type == "short" || type == "int16") return sizeof(std::int16_t);
  if (type == "uint" || type == "uint32") return sizeof(std::uint32_t);
  if (type == "int" || type == "int32") return sizeof(std::int32_t);
  throw std::runtime_error("Unsupported scalar type: " + type);
}

static size_t GetVertexPropertyByteLength(
    const std::vector<PropertyInfo>& vertexProperties) {
  size_t totalLength = 0;
  for (size_t propIndex = 0; propIndex < vertexProperties.size(); ++propIndex) {
    totalLength += GetPropertyByteLength(vertexProperties[propIndex].type);
  }
  return totalLength;
}

static float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

PointCloud PlyLoader::LoadPointCloud(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open PLY file: " + path);
  }

  std::string line;
  std::getline(input, line);
  if (line != "ply") {
    throw std::runtime_error("Invalid PLY header.");
  }

  bool isBinaryLittleEndian = false;
  bool inVertexElement = false;
  size_t vertexCount = 0;
  std::vector<PropertyInfo> vertexProperties;

  while (std::getline(input, line)) {
    if (line == "end_header") {
      break;
    }

    cout << line << endl;

    std::istringstream iss(line);
    std::string token;
    iss >> token;

    if (token == "format") {
      std::string formatName;
      std::string version;
      iss >> formatName >> version;
      if (formatName == "binary_little_endian") {
        isBinaryLittleEndian = true;
      }
    } else if (token == "element") {
      std::string elementName;
      size_t count = 0;
      iss >> elementName >> count;
      inVertexElement = (elementName == "vertex");
      if (inVertexElement) {
        vertexCount = count;
      }
    } else if (token == "property" && inVertexElement) {
      std::string type;
      std::string name;
      iss >> type >> name;

      if (type == "list") {
        throw std::runtime_error(
            "List properties are not supported in vertex element.");
      }

      vertexProperties.push_back({type, name});
    }
  }

  if (!isBinaryLittleEndian) {
    throw std::runtime_error(
        "Only binary_little_endian PLY is supported in this first version.");
  }

  if (vertexCount == 0) {
    throw std::runtime_error("No vertex element found in PLY.");
  }

  size_t vertexPropertyByteLength =
      GetVertexPropertyByteLength(vertexProperties);
  if (vertexPropertyByteLength != sizeof(PlyGaussianVertex)) {
    throw std::runtime_error(
        "Vertex property byte length does not match PlyGaussianVertex size.");
  }

  // 一次性读取整个文件块
  std::vector<PlyGaussianVertex> gaussianCloud(vertexCount);
  size_t vertexBufferByteLength = vertexCount * vertexPropertyByteLength;
  input.read(reinterpret_cast<char*>(gaussianCloud.data()),
             vertexBufferByteLength);
  if (input.gcount() != static_cast<std::streamsize>(vertexBufferByteLength)) {
    throw std::runtime_error("PLY file is truncated or corrupted.");
  }

  PointCloud cloud;
  cloud.points.resize(vertexCount);

  cout << "Vertex count: " << vertexCount << endl;
  cout << "cloud.points.size(): " << cloud.points.size() << endl;
  cout << "cloud.points.capacity(): " << cloud.points.capacity() << endl;

  for (size_t vertex = 0; vertex < vertexCount; ++vertex) {
    cloud.points[vertex].position =
        Eigen::Vector3f(gaussianCloud[vertex].x, gaussianCloud[vertex].y,
                        gaussianCloud[vertex].z);
    cloud.points[vertex].color = DecodeColorFromDC(gaussianCloud[vertex].f_dc);
    cloud.points[vertex].opacity = Sigmoid(gaussianCloud[vertex].opacity);
    cloud.points[vertex].scale =
        Eigen::Vector3f(std::exp(gaussianCloud[vertex].scale[0]),
                        std::exp(gaussianCloud[vertex].scale[1]),
                        std::exp(gaussianCloud[vertex].scale[2]));
    cloud.points[vertex].rotation = Eigen::Vector4f(
        gaussianCloud[vertex].rot[0], gaussianCloud[vertex].rot[1],
        gaussianCloud[vertex].rot[2], gaussianCloud[vertex].rot[3]);
    // cloud.points[vertex].rotation = Eigen::Vector4f(
    //    gaussianCloud[vertex].rot[1], gaussianCloud[vertex].rot[2],
    //    gaussianCloud[vertex].rot[3], gaussianCloud[vertex].rot[0]);

    auto& sh = cloud.points[vertex].sh;
    // 先放 DC
    sh[0] = gaussianCloud[vertex].f_dc[0];
    sh[16] = gaussianCloud[vertex].f_dc[1];
    sh[32] = gaussianCloud[vertex].f_dc[2];
    // 再放 rest
    for (int i = 0; i < 15; ++i) {
      sh[1 + i] = gaussianCloud[vertex].f_rest[i];
      sh[16 + 1 + i] = gaussianCloud[vertex].f_rest[15 + i];
      sh[32 + 1 + i] = gaussianCloud[vertex].f_rest[30 + i];
    }
  }

  return cloud;
}
