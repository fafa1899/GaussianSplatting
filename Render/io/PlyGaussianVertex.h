#pragma once

// 强制 1 字节对齐，确保 sizeof(GaussianVertex) == 236
#pragma pack(push, 1)
struct PlyGaussianVertex {
  float x, y, z;
  float nx, ny, nz;
  float f_dc[3];
  float f_rest[45];
  float opacity;
  float scale[3];
  float rot[4];
};
#pragma pack(pop)

// 编译期断言：确保结构体大小严格等于 62 * 4 = 248 字节
static_assert(sizeof(PlyGaussianVertex) == 62 * sizeof(float),
              "GaussianVertex size mismatch!");