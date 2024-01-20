find_package(rostest REQUIRED)
catkin_add_gtest(test_${PROJECT_NAME} main.cpp test_embedding_distances.cpp)
target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME})
