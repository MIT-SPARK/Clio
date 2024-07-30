cmake_minimum_required(VERSION 3.14)
project(clio VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

option(CLIO_ENABLE_TESTS "Build unit tests" OFF)
option(BUILD_SHARED_LIBS "Build shared libs" ON)

find_package(hydra REQUIRED)
find_package(khronos REQUIRED)

include(GNUInstallDirs)

add_library(
  ${PROJECT_NAME}
  src/agglomerative_clustering.cpp
  src/clustering_workspace.cpp
  src/ib_utils.cpp
  src/ib_edge_selector.cpp
  src/object_update_functor.cpp
  src/probability_utilities.cpp
  src/region_update_functor.cpp
)
target_include_directories(
  ${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>
                         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(${PROJECT_NAME} PUBLIC hydra::hydra khronos::khronos)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(clio::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

if(CLIO_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT ${PROJECT_NAME}-targets FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE clio:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clio
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/clioConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
write_basic_package_version_file(
  ${PROJECT_NAME}ConfigVersion.cmake VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
