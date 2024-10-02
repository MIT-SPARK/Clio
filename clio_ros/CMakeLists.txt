cmake_minimum_required(VERSION 3.14)
project(clio_ros)

find_package(catkin REQUIRED)

catkin_package(
  CATKIN_DEPENDS
  DEPENDS
  INCLUDE_DIRS
  LIBRARIES
)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
