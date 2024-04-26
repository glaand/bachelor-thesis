project(cfd_2d)

find_package(Eigen3 REQUIRED COMPONENTS CXX)
find_package(Torch REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set(CFD_2D_SOURCES
    src/cfd_params.cpp
    src/cfd.cpp
    src/vtk.cpp
    src/multigrid.cpp
    src/solvers/jacobi.cpp
    src/solvers/multigrid_jacobi.cpp
    src/solvers/cg.cpp
    src/solvers/apcg.cpp
    src/solvers/jpcg.cpp
    src/solvers/mgpcg.cpp
    src/solvers/sd.cpp
    src/solvers/mgpsd.cpp
    src/solvers/ml.cpp
    src/solvers/richardson.cpp
)

# Compile the CFD library
add_library(cfd_2d STATIC ${CFD_2D_SOURCES})
target_compile_options(cfd_2d PRIVATE)
target_include_directories(cfd_2d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(
    cfd_2d 
    Eigen3::Eigen 
    kernel 
    VTK::CommonDataModel
    VTK::CommonCore
    VTK::CommonMath
    VTK::CommonMisc
    VTK::CommonSystem
    VTK::CommonTransforms
    VTK::IOLegacy
    VTK::IOParallelXML
    VTK::IOXML
    VTK::IOXMLParser
    VTK::ParallelCore
    VTK::FiltersCore
    VTK::vtksys
    torch
    OpenMP::OpenMP_CXX
)

# Compile CFD Programs 
## Lid Driven Cavity 2D
add_executable(lid_driven_cavity_2d prog/lid_driven_cavity_2d.cpp)
target_compile_options(lid_driven_cavity_2d PRIVATE -fPIC)
target_include_directories(lid_driven_cavity_2d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lid_driven_cavity_2d cfd_2d)

# Flow over step 2D
add_executable(flow_over_step_2d prog/flow_over_step_2d.cpp)
target_compile_options(flow_over_step_2d PRIVATE -fPIC)
target_include_directories(flow_over_step_2d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(flow_over_step_2d cfd_2d)

# Karman Vortex Street 2D
add_executable(karman_vortex_street_2d prog/karman_vortex_street_2d.cpp)
target_compile_options(karman_vortex_street_2d PRIVATE -fPIC)
target_include_directories(karman_vortex_street_2d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(karman_vortex_street_2d cfd_2d)

# Plane Shear Flow 2D
add_executable(plane_shear_flow_2d prog/plane_shear_flow_2d.cpp)
target_compile_options(plane_shear_flow_2d PRIVATE -fPIC)
target_include_directories(plane_shear_flow_2d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(plane_shear_flow_2d cfd_2d)

# Tests
add_executable(test_multigrid tests/test_multigrid.cpp)
target_include_directories(test_multigrid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_multigrid PRIVATE cfd_2d)