# 3DGS CPU Offline Renderer Implementation Plan

## Goal

Implement a pure CPU offline renderer for 3D Gaussian Splatting in C/C++ under `D:\3DGS\Unity3DGS\GaussianSplatting\Render`.

Constraints:

- Do not consider real-time rendering.
- Do not consider integration with any rendering pipeline.
- Use C/C++.
- Use Eigen for matrix and vector math.
- Use OpenCV for image output and simple image processing.
- Use CPU algorithms first.
- Replace CUDA-style parallelism with OpenMP where useful.

Primary target:

- Render a single image from a specified camera pose.
- Progressively approach the visual result of the official `gaussian-splatting` implementation.

Reference repositories:

- Official implementation: `D:\3DGS\Unity3DGS\gaussian-splatting`
- Cesium reference: `D:\3DGS\Unity3DGS\cesium`
- Cesium WASM splat utilities: `D:\3DGS\Unity3DGS\cesium-wasm-utils`

Primary test data:

- `D:\3DGS\Unity3DGS\models\bicycle\point_cloud\iteration_30000\point_cloud.ply`

## Development Strategy

Use a coarse-to-fine iterative strategy:

1. Always keep the renderer runnable.
2. Ensure every stage produces a visible image.
3. Add one major feature at a time.
4. Validate output after each stage before moving on.
5. Prioritize correctness first, performance second.

This project should evolve from:

- point cloud projection
- to sorted splats
- to opacity blending
- to scale-aware splats
- to anisotropic Gaussian splatting
- to SH-based view-dependent color
- to official-result alignment

## Final Scope

The final renderer should support:

- Loading Gaussian parameters from official `point_cloud.ply`
- Camera-based offline rendering
- CPU rasterization of 3D Gaussians
- Screen-space Gaussian evaluation
- Front-to-back alpha compositing
- SH-based color evaluation
- PNG output

The final renderer does not need to support:

- training
- backward pass
- interactive viewer
- GPU path
- Unity rendering pipeline adaptation

## Key Technical Risks

### 1. Camera and matrix convention mismatch

The official implementation uses specific matrix layout and multiplication conventions. Small mistakes here can produce:

- flipped images
- wrong camera direction
- incorrect depth ordering
- distorted projection

### 2. 3D-to-2D covariance projection details

The most important math in 3DGS lies in projecting 3D covariance into a 2D screen-space Gaussian ellipse. This must be aligned closely with the official implementation.

### 3. Ordering and blending

Visual similarity depends heavily on:

- depth ordering
- alpha compositing order
- opacity activation
- early termination thresholds

### 4. SH color evaluation details

The SH evaluation contains several small but important details such as:

- basis ordering
- view direction convention
- `+0.5` offset
- clamping behavior

### 5. CPU performance

Performance will be much slower than the CUDA version. This is acceptable for offline rendering, but the implementation should still be structured for later optimization with OpenMP and spatial batching.

## Incremental Phases

## Phase 0 - Project Skeleton

### Objective

Create a minimal offline rendering executable with clear module boundaries.

### Tasks

- Set up CMake for:
  - Eigen
  - OpenCV
  - OpenMP
- Create base source layout
- Define core data structures
- Add command-line entry point
- Add image save path handling
- Add logging or simple console progress output

### Suggested Output

- A test image or blank image saved successfully

### Acceptance Criteria

- Project builds successfully
- Executable runs successfully
- Image output path works

## Phase 1 - Basic Point Cloud Rendering

### Objective

Render the Gaussian centers as ordinary projected 3D points.

### Tasks

- Load `xyz` from `point_cloud.ply`
- Ignore scale, rotation, opacity, and SH for now
- Implement:
  - world to view transform
  - view to clip transform
  - NDC to pixel conversion
- Rasterize each point as:
  - one pixel
  - or a fixed-size small disk
- Use a simple constant color first

### Purpose

Validate camera math and image-space mapping before adding Gaussian-specific behavior.

### Acceptance Criteria

- A recognizable bicycle silhouette appears
- The scene is not mirrored incorrectly
- Major projection bugs are eliminated

## Phase 2 - Depth-Sorted Point Splat Rendering

### Objective

Introduce depth ordering and simple splat overlap.

### Tasks

- Compute view-space depth per point
- Sort points by depth
- Replace single-pixel plotting with small circular splats
- Add simple front-to-back or back-to-front overlap handling

### Purpose

Validate visibility ordering before introducing opacity and Gaussian kernels.

### Acceptance Criteria

- Foreground points consistently cover background points
- Shape appears more solid than plain point cloud rendering

## Phase 3 - Opacity-Based Compositing

### Objective

Introduce the official opacity parameter and alpha blending behavior.

### Tasks

- Load `opacity` from PLY
- Apply the same activation convention as the official implementation
- Implement front-to-back alpha compositing
- Add configurable background color

### Purpose

Move from solid point splats toward translucent accumulation behavior used by 3DGS.

### Acceptance Criteria

- Transparent accumulation behaves stably
- Object edges become softer and more continuous
- Background blending behaves as expected

## Phase 4 - Scale-Aware Isotropic Splatting

### Objective

Use Gaussian scale values to control projected splat size.

### Tasks

- Load `scale_0`, `scale_1`, `scale_2`
- First simplify them into an isotropic screen-space radius
- Replace fixed-size splats with per-Gaussian size
- Continue using circular kernels

### Purpose

Bridge the gap between naive point splats and full anisotropic Gaussian rendering.

### Acceptance Criteria

- Larger Gaussians visibly cover larger screen areas
- Sparse regions become more continuous
- Rendering looks more like splatting than point drawing

## Phase 5 - Full 3D Covariance and Screen-Space Elliptical Gaussian

### Objective

Implement the core 3DGS rendering math.

### Tasks

- Load `rotation`
- Reconstruct 3D covariance from:
  - scale
  - quaternion rotation
- Project 3D covariance to 2D covariance using official-aligned math
- Convert projected covariance into:
  - conic form
  - ellipse bounds
- Rasterize elliptical Gaussian splats over local pixel bounding boxes

### Purpose

This is the key stage where the renderer becomes a true Gaussian splatting renderer instead of a point renderer.

### Acceptance Criteria

- Splat shape is anisotropic and directionally correct
- Surface appearance becomes smoother and more coherent
- Visual quality approaches the expected 3DGS look

## Phase 6 - SH-Based View-Dependent Color

### Objective

Support official spherical harmonics color evaluation.

### Tasks

- Load:
  - `f_dc_*`
  - `f_rest_*`
- Implement SH basis evaluation
- Compute color from camera-relative viewing direction
- Align with official behavior:
  - basis order
  - `+0.5`
  - clamp to non-negative

### Purpose

Add the view-dependent appearance that strongly affects the final 3DGS look.

### Acceptance Criteria

- Rendered color changes correctly with viewpoint
- Image becomes visibly closer to the official renderer

## Phase 7 - Official Alignment and CPU Optimization

### Objective

Reduce remaining visual differences and improve CPU runtime.

### Correctness Tasks

- Align near-plane culling behavior
- Align covariance bias / low-pass stabilization
- Align antialiasing handling
- Align alpha threshold and early-stop behavior
- Align pixel-center convention
- Compare against official render output for a fixed camera

### Optimization Tasks

- Add OpenMP to Gaussian preprocessing
- Add OpenMP to raster stage where safe
- Add bounding-box clipping improvements
- Reduce repeated matrix computations
- Consider tile-based CPU processing if needed

### Acceptance Criteria

- Visual similarity to official output is high
- Offline render time is acceptable for single-frame generation

## Suggested Implementation Order

Strict recommended order:

1. Project setup and image output
2. Point projection
3. Depth-sorted splats
4. Opacity compositing
5. Scale-driven radius
6. Full covariance and elliptical Gaussian rasterization
7. SH color
8. Official alignment
9. Performance optimization

This order is important because it ensures:

- every stage is debuggable
- every stage produces an image
- problems can be isolated quickly

## Suggested Code Organization

Recommended source modules:

- `main.cpp`
- `AppConfig.h/.cpp`
- `GaussianData.h/.cpp`
- `PlyLoader.h/.cpp`
- `Camera.h/.cpp`
- `MathUtils.h/.cpp`
- `ShEvaluator.h/.cpp`
- `ImageBuffer.h/.cpp`
- `PointRasterizer.h/.cpp`
- `GaussianProjector.h/.cpp`
- `GaussianRasterizer.h/.cpp`
- `Renderer.h/.cpp`

Possible data structures:

- `GaussianPoint`
  - position
  - opacity
  - scale
  - rotation
  - SH coefficients

- `CameraParameters`
  - width
  - height
  - fovx
  - fovy
  - view matrix
  - projection matrix
  - camera center

- `ProjectedGaussian`
  - screen center
  - depth
  - radius
  - covariance2D
  - conic
  - color
  - alpha

## Reference Mapping to Official Implementation

Important official files to mirror conceptually:

- `gaussian-splatting/gaussian_renderer/__init__.py`
  - high-level render flow

- `gaussian-splatting/scene/gaussian_model.py`
  - PLY field layout
  - scale / rotation / opacity conventions

- `gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`
  - SH to RGB
  - covariance math
  - preprocess behavior
  - alpha compositing

- `gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h`
  - matrix helpers
  - NDC to pixel conversion
  - frustum logic

Notes:

- The GPU tile binning logic in the official implementation is primarily a performance strategy.
- The CPU first version does not need to replicate the exact CUDA work scheduling structure.
- The math and compositing order matter more than the batching structure.

## Validation Plan

At each phase, save output images to a stable directory and compare visually.

Suggested validation method:

1. Fix one camera pose.
2. Render one output image at each phase.
3. Keep all outputs for comparison.
4. After Phase 5 and Phase 6, compare against official output from the same camera.

Suggested output folder:

- `D:\3DGS\Unity3DGS\GaussianSplatting\Render\outputs`

Suggested image naming:

- `phase1_points.png`
- `phase2_sorted_splats.png`
- `phase3_opacity.png`
- `phase4_scaled_splats.png`
- `phase5_gaussian_ellipse.png`
- `phase6_sh_color.png`
- `phase7_aligned.png`

## Milestone Summary

### Milestone A

- Project builds
- PLY loads
- Basic point cloud image renders

### Milestone B

- Depth ordering and opacity work
- Splat-based rendering becomes stable

### Milestone C

- Full anisotropic Gaussian math works
- Output visibly resembles 3DGS

### Milestone D

- SH color works
- Output approaches official renderer

### Milestone E

- OpenMP optimization added
- Offline rendering is practical for repeated use

## Recommended Near-Term Next Step

Start with:

- Phase 0
- Phase 1

Immediate implementation target:

- Build a command-line program that loads `point_cloud.ply`
- Construct a temporary test camera
- Render a simple projected point cloud image
- Save it as PNG

This gives the first visible result quickly and creates a stable base for all later stages.
