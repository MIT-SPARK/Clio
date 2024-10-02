# Clio

This repository contains the code for *Clio: Real-time Task-Driven Open-Set 3D Scene Graphs*.

We have included code here to quickly get you up and running an offline version of Clio (Clio-batch) and visualize results.
We will release the real-time version of Clio (Clio-online) that can process incoming images+poses on October 7th!

![Clio](./assets/overview-figure.jpg)

Clio is a novel approach for building task-driven 3D scene graphs in real-time with open-set semantics. We draw inspiration from the classical Information Bottleneck principle to form task- relevant clusters of object primitives given a set of natural language tasks — such as ''Read brown textbook'' — and by clustering the scene into task-relevant semantic regions such as “Kitchenette” or “Workspace”. The map defines objects and regions at the correct semantic granularity to support tasks relevant for an agent.

# Table of Contents
* [Datasets](#Datasets)
* [Setup](#Setup)
* [Evaluation](#Evaluation)
* [Running Clio-Online with ROS](#running-clio-online)
* [Paper](#Paper)
* [Video](#Video)

# Datasets
- Our custom datasets for the Office, Apartment, Cubicle, and Building scenes are available for download at [Dropbox-Link-Dataset](https://www.dropbox.com/scl/fo/5bkv8rsa2xvwmvom6bmza/AOc8VW71kuZCgQjcw_REbWA?rlkey=wx1njghufcxconm1znidc1hgw&st=c809h8h3&dl=0). Each scene contains RGB images, depth images, a rosbag containing the RGB and Depth images along with poses, and the list of tasks with ground truth object labels that was used in our paper. Each scene except the Building also contains a COLMAP dense reconstruction which can optionally be used to separately get a dense mesh view of the scene.
 - The task list is stored in a yaml file whose keys are the task and values are the ground truth oriented bounding boxes for the relevant objects. The folder structure is:

 ```
 clio_datasets
├── apartment
│   ├── apartment.bag
│   ├── database.db
│   ├── dense
|       ├── fused.ply
|       |── meshed-poisson.ply
│   ├── depth
│   ├── images
│   ├── rooms_apartment.yaml
│   ├── sparse
│   └── tasks_apartment.yaml
├── building
│   ├── ...
├── cubicle
│   ├── ...
├── office
│   ├── ...
 ```

## Pre-built Scene Graphs
 - Pre-built scene graph files can be downloaded from [Dropbox-Link-Log](https://www.dropbox.com/scl/fo/dmft0x6o0joxjxyy2q1qz/AIbgXuR4Y0nxMgsq1VtlWak?rlkey=g3orpk2itv1eyg0r1h7mpfgtt&st=po6btms5&dl=0). These will be needed if you want to test out Clio-batch. These dsg files already contain the 3D object primitives with corresponding mesh and semantic embedding vector which Clio-batch will use to form task-relevant objects.

## Pre-generating Open-set Semantics for a Scene

> **Warning** :warning: </br>
> This requires the [semantic_inference](https://github.com/MIT-SPARK/semantic_inference) package and ROS, which is installed by default if you follow the normal setup guide for Clio.
> It will not be installed if you have only followed the setup for Clio-batch and the evaluation code.

It may be convenient to generate the open-set segmentation and CLIP embeddings for a scene before running the Clio pipeline.
You can run the following commands for any of the scenes, substituting the appropriate path to the rosbag for the scene.
First, source your `semantic_inference` environment and change to the directory containing the datasets if you haven't already
```bash
source ~/environments/semantic_inference_ros/bin/activate
cd /path/to/clio/datasets
```
and then using the apartment scene as an example, run
```bash
rosrun semantic_inference_ros make_rosbag --clip-vec --copy \
    apartment/apartment.bag -o apartment/apartment_with_semantics.bag \
    /dominic/forward/color/image_raw:/dominic/forward/semantic/image_raw
```
to create a new bag, `apartment_with_semantics.bag` that contains the original contents of `apartment.bag` along with the open-set segmentation (under the `/dominic/forward/semantic/image_raw` topic).

# Setup

We recommend that everyone setup Clio by utilizing ROS.
You can follow the instructions [here](http://wiki.ros.org/ROS/Installation) to install ROS if you haven't already.
If you just want to install Clio-Batch and Clio-Eval and want to avoid installing ROS, you can skip to [these](#installing-without-ros) instructions instead.

### Installing with ROS

<details open>

<summary><b>Initial Requirements</b></summary>

Install the following general requirements via
```
sudo apt install python3-rosdep python3-catkin-tools python3-vcstool python3-virtualenv
```

If you haven't set up rosdep yet run
```
sudo rosdep init
rosdep update
```

</details>

<details open>

<summary><b>Getting and Building Clio</b></summary>

To clone and build Clio, run the following
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
catkin config -DCMAKE_BUILD_TYPE=Release

cd src
git clone git@github.com:MIT-SPARK/Clio.git clio --recursive
vcs import . < clio/install/clio.rosinstall
rosdep install --from-paths . --ignore-src -r -y

cd ..
catkin build
```

> **Note**</br>
> For the rest of these instructions, we assume that you set up your catkin workspace at `~/catkin_ws/src`.
> If you used a different workspace path, you should substitute where appropriate.

</details>

<details open>

<summary><b>Setting up Python Code</b></summary>

Make a virtual environment
```bash
python3 -m virtualenv --system-site-packages -p /usr/bin/python3 ~/environments/clio
```
and then install
```bash
source ~/environments/clio/bin/activate
pip install -e ~/catkin_ws/src/clio
```

> **Warning** :warning:</br>
> Both `--system-site-packages` when creating the environment and `-e` when installing Clio are required.

To run open-set segmentation, you'll also have to install `semantic_inference`
```bash
source ~/environments/clio/bin/activate  # if you haven't already
pip install ~/catkin_ws/src/semantic_inference/semantic_inference[openset]
```

</details>

### Installing without ROS

First, setup a virtual environment
```bash
python3 -m virtualenv -p /usr/bin/python3 --download ~/environments/clio
```

Then, clone and install Clio
```bash
source ~/environments/clio/bin/activate
git clone https://github.com/MIT-SPARK/Clio.git clio --recursive
pip install -e clio
```

> **Note**</br>
> If you forgot to clone with `--recursive` you can run `git submodule update --init --recursive` instead.

# Evaluation

To run Clio on our datasets and metrics we provide evaluation scripts that can automatically run different configurations of Clio and baseline methods on the datasets.
We provide a batch version of Clio referred to as Clio-batch which is based in python and an incremental version which runs in realtime referred to as Clio-online which is built primarily in c++.

## Clio-Batch

Clio-batch will perform task-oriented clustering on a pre-build scene graph of object primitives.
To try out Clio, you can run
```
cd clio
python clio_batch/object_cluster.py <path-to-scene-graph-json> <path-to-task-yaml> <path-to-output-folder> <path-to-cluster-config>
```
For example, running Clio just on the pre-built office scene graph (assuming you have made an empty folder first called clio_output):
```
python clio_batch/object_cluster.py ~/clio_logs/office/trial_0/large_fine_ps/dsg.json ~/clio_datasets/office/tasks_office.yaml ~/clio_output clio_eval/experiments/configs/cluster/large_delta_00001.yaml
```

To quickly visualize results, you can visualize the estimated Clio bounding boxes (in green) and ground truth bounding boxes (in black) over a COLMAP mesh. To do this on the office scene, run

```
python clio_eval/visualize_objects.py ~/clio_output/clio_dsg.json ~/clio_datasets/office/tasks_office.yaml  ~/clio_datasets/office/dense/meshed-poisson.ply office
```

This will generate a 3D viewer such as this:

![cluster](./assets/sample_clustered.png)

Change to either apartment or cubicle with the following to generate the visualization on those datasets:

```
python clio_eval/visualize_objects.py clio_dsg.json ~/clio_datasets/apartment/tasks_apartment.yaml ~/clio_datasets/apartment/dense/meshed-poisson.ply apartment

python clio_eval/visualize_objects.py clio_dsg.json ~/clio_datasets/cubicle/tasks_cubicle.yaml ~/clio_datasets/cubicle/dense/meshed-poisson.ply cubicle
```

### Running the Full Evaluation
A config file for using CLIP model ViT-L/14 used in the main results of the paper is provided in clio_eval/experiments/configs/ablations/3d_clustering.yaml and a config file for Open-CLIP model ViT-H-14
used to produce results in the Appendix of the paper is provided in clio_eval/experiments/configs/ablations/3d_clustering_oc.yaml. The yaml files allow for running on one or on multiple datasets at a time.

First, set paths to your pre-built scene graph files and yaml files containing the ground truth objects and task labels in 3d_clustering.yaml and 3d_clustering_oc.yaml

To run Clio-batch and run evaluation scripts, run:

```
python clio_eval/experiments/run_3d_object_ablations.py  clio_eval/experiments/configs/ablations/<3d_clustering.yaml/3d_clustering_oc.yaml>
```

The provided results will include for each dataset:
Clio Primitives (labeled as <dataset>_fine)
Khronos (labeled as <dataset>_khronos)
Khronos thresholded (labeled as <dataset>_khronos_thres)
Clio batch (labeled as <dataset>_d<IB delta stop value>)

By default, the config files are set to run on all three of the office, apartment, and cubicle datasets. Results from clustering will be saved in a newly created dsg files for each dataset.
If you already ran Clio-batch and have made these dsg files, the above code will skip running clio-batch and just run the evals.


# Running Clio-Online

To run Clio-Online on one of the provided datasets (see [#Datasets] for more details), first appropriate source your catkin workspace and python environment
```bash
source ~/catkin_ws/devel/setup.bash
source ~/environments/clio/bin/activate
```

Then, start Clio
```bash
roslaunch clio_ros realsense.launch object_tasks_file:=/path/to/datasets/apartment/tasks_apartment.yaml
```
If you want to use pre-generated segmentations and semantics instead, you can start Clio-Online with the following
```bash
roslaunch clio_ros realsense.launch run_segmentation:=false object_tasks_file:=/path/to/datasets/apartment/tasks_apartment.yaml
```

In a separate terminal, start the corresponding rosbag for the scene. For the apartment
```bash
rosbag play /path/to/datasets/apartment/apartment.bag --clock
```

# Paper
If you find this useful for your research, please consider citing our paper:
* Dominic Maggio, Yun Chang, Nathan Hughes, Matthew Trang, Dan Griffith, Carlyn Dougherty, Eric Cristofalo, Lukas Schmid, Luca Carlone, "**Clio: Real-time Task-Driven Open-Set 3D Scene Graphs**", in IEEE Robotics and Automation Letters, vol. 9, no. 10, pp. 8921-8928, Oct. 2024. [ArXiv-Link](http://arxiv.org/abs/2404.13696)

  ```bibtex
   @ARTICLE{Maggio2024Clio,
      title={Clio: Real-time Task-Driven Open-Set 3D Scene Graphs},
      author={Maggio, Dominic and Chang, Yun and Hughes, Nathan and Trang, Matthew and Griffith, Dan and Dougherty, Carlyn and Cristofalo, Eric and Schmid, Lukas and Carlone, Luca},
      journal={IEEE Robotics and Automation Letters},
      year={2024},
      volume={9},
      number={10},
      pages={8921-8928},
      doi={10.1109/LRA.2024.3451395}
   }
  ```

# Video
An overview of Clio is available on [YouTube](https://www.youtube.com/watch?v=m-HJO10qhSQ)
