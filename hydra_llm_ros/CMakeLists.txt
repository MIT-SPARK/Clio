cmake_minimum_required(VERSION 3.14)
project(hydra_llm_ros)

add_compile_options(-Wall -Wextra)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# TODO(nathan) we should inherit this, but catkin is broken for pure-cmake?
find_package(hydra REQUIRED)
find_package(catkin REQUIRED COMPONENTS hydra_llm hydra_ros khronos_ros llm roscpp)

catkin_package(
  CATKIN_DEPENDS hydra_llm hydra_ros khronos_ros llm roscpp
  DEPENDS hydra
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

add_library(
  ${PROJECT_NAME}
  src/active_window_module.cpp
  src/category_legend.cpp
  src/colormap_legend.cpp
  src/graph_wrappers.cpp
  src/gt_publisher.cpp
  src/hydra_llm_pipeline.cpp
  src/interlayer_plugin.cpp
  src/khronos_input_module.cpp
  src/llm_frontend.cpp
  src/llm_layer_color_functor.cpp
  src/llm_layer_plugin.cpp
  src/marker_tracker.cpp
  src/node_filter.cpp
  src/ros_embedding_group.cpp
  src/task_information.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC include ${catkin_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${catkin_LIBRARIES} hydra::hydra)

add_executable(hydra_llm_node app/hydra_llm_node.cpp)
target_link_libraries(hydra_llm_node PUBLIC ${PROJECT_NAME})

# TODO(nathan) literally a copy of the hydra node to have new plugin registrations in
# the same shared object
add_executable(hydra_llm_visualizer_node app/hydra_llm_visualizer_node.cpp)
target_link_libraries(hydra_llm_visualizer_node PUBLIC ${PROJECT_NAME})

add_executable(clustering_visualizer app/clustering_visualizer.cpp)
target_link_libraries(clustering_visualizer PUBLIC ${PROJECT_NAME})

install(
  TARGETS ${PROJECT_NAME} hydra_llm_node hydra_llm_visualizer_node clustering_visualizer
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
