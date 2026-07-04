# CMakeList.txt: Render 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#

# 判断编译器类型
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    message(">> using Clang")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message(">> using GCC")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message(">> using Intel C++")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    message(">> using Visual Studio C++")
    add_compile_options("/utf-8")
else()
    message(">> unknow compiler.")
endif()

set(OpenCV_ARCH "x64")
set(OpenCV_RUNTIME "vc16") 

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED CONFIG)

# 将源代码添加到此项目的可执行文件。
add_executable (Render "Main.cpp" "app/App.h" "app/App.cpp" "io/PlyLoader.h" "scene/PointCloud.h" "io/PlyLoader.cpp" "io/PlyGaussianVertex.h" "scene/Camera.h" "scene/Camera.cpp" "math/Transform.h" "math/Transform.cpp" "render/PointRenderer.h" "render/PointRenderer.cpp")

target_include_directories(Render PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(Render PRIVATE Eigen3::Eigen)
target_link_libraries(Render PRIVATE opencv_world)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Render PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
