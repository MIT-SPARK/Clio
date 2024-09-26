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
* [Running Clio-Online with ROS](#running-clio-online-with-ros)
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

# Setup
```bash
git clone https://github.com/MIT-SPARK/Clio.git clio --recursive
```

Make sure to clone with `--recursive`! Alternatively, run
```bash
git submodule update --init --recursive
```
from the path to the repo once cloned.

Make a virtual environment.
```bash
python3 -m virtualenv -p /usr/bin/python3 --download ~/py3_clio # or whatever you want to name your virtualenv
```

Then
```
source ~/py3_clio/bin/activate # activate virtualenv

cd clio
pip install -e .
```

Clone Spark-DSG from https://github.com/MIT-SPARK/Spark-DSG/tree/develop. Make sure you are on the develop branch. To install, inside Spark-DSG run: 
```
cd ..
git clone https://github.com/MIT-SPARK/Spark-DSG.git --branch develop spark_dsg
cd spark_dsg
pip install -e .
```

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


## Clio-Online

- Coming on October 7th

<!-- Clio-online results can be generated by running:

```
python3 clio_eval/experiments/run_ablations.py --configs clio_eval/experiments/configs/ablations/realsense.yaml --overwrite --visualize
```

To run metrics on the generated results, run:

```
python3 clio_eval/experiments/evaluate_ablations.py --configs clio_eval/experiments/configs/ablations/realsense.yaml
```

online_segmentation=false will look for pre-generated bags containing Segmentation and CLIP vectors. Online_segmentation=true will run full the Clio pipeline in realtime. -->

## Places Evaluation
- Coming on October 7th

# Running Clio-Online with ROS

- Coming on October 7th

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
