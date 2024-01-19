# Hydra-LLM (working title)


## Installation

First, build the ROS code:

```bash
sudo apt install libprotobuf-dev protobuf-compiler

mkdir -p catkin_ws/src
cd catkin_ws
catkin init
catkin config -DCMAKE_BUILD_TYPE=Release -DGTSAM_TANGENT_PREINTEGRATION=OFF \
              -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF -DGTSAM_USE_SYSTEM_EIGEN=ON
catkin config --skiplist hdf5_map_io mesh_msgs_hdf5 label_manager mesh_tools \
                         rviz_map_plugin minkindr_python

cd src
git clone git@github.mit.edu:SPARK/Hydra-LLM.git hydra_llm
vcs import . < hydra_llm/install/hydra_llm.rosinstall

rosdep install --from-paths . --ignore-src -r -y

cd ..
catkin build
```

Then, follow the install instructions [here](https://github.mit.edu/drmaggio/LLM-Graphs#running-with-ros-directly--running-places-clip-extraction) to set up a virtual environment for CLIP.

## Run with uHumans2

Run the following:
```bash
source path/to/virtual/environment/bin/activate
roslaunch hydra_llm_ros uhumans2.launch tasks:="kitchen*bathroom*office*hallway*bedroom*lounge*stairwell*cubicle*doorway*conference room"
```

The tasks can be set to whatever you want (asterisks separate each task).
