project(cfd_3d)

find_package(Eigen3 REQUIRED COMPONENTS CXX)

set(CFD_3d_SOURCES
    src/cfd_params.cpp
    src/cfd.cpp
    src/vtk.cpp
    src/multigrid.cpp
    src/solvers/jacobi.cpp
    src/solvers/multigrid_jacobi.cpp
    src/solvers/conjugate_gradient.cpp
    src/solvers/multigrid_pcg.cpp
)

# Compile the CFD library
add_library(cfd_3d STATIC ${CFD_3d_SOURCES})
target_compile_options(cfd_3d PRIVATE)
target_include_directories(cfd_3d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(
    cfd_3d 
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
)

# Compile CFD Programs 
## Lid Driven Cavity 3d
add_executable(lid_driven_cavity_3d prog/lid_driven_cavity_3d.cpp)
target_compile_options(lid_driven_cavity_3d PRIVATE -fPIC)
target_include_directories(lid_driven_cavity_3d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lid_driven_cavity_3d cfd_3d)

# Flow over step 3d
#add_executable(flow_over_step_3d prog/flow_over_step_3d.cpp)
#target_compile_options(flow_over_step_3d PRIVATE -fPIC)
#target_include_directories(flow_over_step_3d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
#target_link_libraries(flow_over_step_3d cfd_3d)

# Karman Vortex Street 3d
add_executable(karman_vortex_street_3d prog/karman_vortex_street_3d.cpp)
target_compile_options(karman_vortex_street_3d PRIVATE -fPIC)
target_include_directories(karman_vortex_street_3d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(karman_vortex_street_3d cfd_3d)

# Plane Shear Flow 3d
#add_executable(plane_shear_flow_3d prog/plane_shear_flow_3d.cpp)
#target_compile_options(plane_shear_flow_3d PRIVATE -fPIC)
#target_include_directories(plane_shear_flow_3d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
#target_link_libraries(plane_shear_flow_3d cfd_3d)
