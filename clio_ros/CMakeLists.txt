cmake_minimum_required(VERSION 3.14)
project(clio_ros)

add_compile_options(-Wall -Wextra)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(gflags REQUIRED)
find_package(clio REQUIRED)
find_package(catkin REQUIRED COMPONENTS hydra_ros hydra_visualizer khronos_ros roscpp)

catkin_package(
  CATKIN_DEPENDS hydra_ros hydra_visualizer khronos_ros roscpp
  DEPENDS clio
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

add_library(
  ${PROJECT_NAME} src/category_legend.cpp src/colormap_legend.cpp src/clio_pipeline.cpp
                  src/llm_frontend.cpp src/task_information.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC include ${catkin_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${catkin_LIBRARIES} clio::clio)

add_executable(${PROJECT_NAME}_node app/clio_node.cpp)
target_link_libraries(${PROJECT_NAME}_node PUBLIC ${PROJECT_NAME} gflags)

install(
  TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
