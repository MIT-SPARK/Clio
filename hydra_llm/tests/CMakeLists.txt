find_package(rostest REQUIRED)
catkin_add_gtest(
  test_${PROJECT_NAME}
  main.cpp
  src/utilities.cpp
  test_agglomerative_clustering.cpp
  test_clustering_workspace.cpp
  test_embedding_distances.cpp
  test_ib_edge_selector.cpp
  test_object_update_functor.cpp
  test_probability_utilities.cpp
)
target_include_directories(test_${PROJECT_NAME} PUBLIC include)
target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME})
