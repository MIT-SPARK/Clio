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

## Running Place Ablations

Almost everything can be found in this script [here](hydra_llm_ros/app/run_place_ablations)

Both `hydra` and `hydra_ros` should be on `llm/feature/places_experiments`. `llm_graphs` should be on `feature/places_experiments` if it isn't merged yet.

I typically have a setup where I have:

```
/path/to/llm_place_results
    data/  # preprocessed data
    gt_artifacts/  # reconstructed meshes and tsdfs
    results_2024_04_07/  # ablation run data
    results_2024_04_08/  # ablation run data
    ...
```

### Preparing data

To prepare new rosbags (the script defaults to bags for ViT-B/32, ViT-L/14, and ViT-H/14):
```
rosrun hydra_llm_ros run_place_ablations prep path/to/desired/output /path/to/bag1 /path/to/bag2 ... -a large
```

However, pepared bags live [here](https://drive.google.com/drive/folders/1vlt4hLqzEgAzs0IROsNZrDLB8DDAo6vL?usp=sharing)

### Making the ground truth

To make the ground-truth, run the following:
```
rosrun hydra_llm_ros run_place_ablations gt path/to/desired/output /path/to/bag1 /path/to/bag2 -v 0.1
```

### Running ablations

To get run the actual pipelines, run the following:
```
rosrun hydra_llm_ros run_place_ablations run path/to/desired/output /path/to/prepped/bag1 /path/to/prepped/bag2 -t NUM_TRIALS
```

### Evaluating ablations

Running the following to evaluate the precision and recall of the rooms:
```
rosrun hydra_llm_ros run_place_ablations eval path/to/results /path/to/gt
```

It defaults to using 8 processes which seems okay, but you can decrease or increase via `-t NUM_DESIRED_PROCESSES`

### Showing / Printing Results

You can run either the `show` or `table` subcommand
```
rosrun hydra_llm_ros run_place_ablations show path/to/results
```
