# Clio

This repository contains the code for *Clio: Real-time Task-Driven Open-Set 3D Scene Graphs*. 

![Clio](./assets/overview-figure.jpg)

Clio is a novel approach for building task-driven 3D scene graphs in real-time with open-set semantics. We draw inspiration from the classical Information Bottleneck principle to form task- relevant clusters of object primitives given a set of natural language tasks — such as ''Read brown textbook'' — and by clustering the scene into task-relevant semantic regions such as “Kitchenette” or “Workspace”. The map defines objects and regions at the correct semantic granularity to support tasks relevant for an agent.

# Table of Contents
* [Datasets](#Datasets)
* [Setup](#Setup)
* [Evaluation](#Evaluation)
* [Paper](#Paper)
* [Video](#Video)      

# Datasets
- Our custom datasets for the Office, Apartment, Cubicle, and Building scenes are available for download at [Dropbox-Link](https://github.com/MIT-SPARK/Clio). Each scene contains RGB images, depth images, a rosbag containing the RGB and Depth images along with poses, and the list of tasks with ground truth object labels that was used in our paper. Each scene except the Building also contains a COLMAP dense reconstruction which can optionally be used to separately get a dense mesh view of the scene. 
 - The task list is stored in a yaml file whose keys are the task and values are the ground truth oriented bounding boxes for the relevant objects.
 - Pre-build scene graphs, TODO Dominic add description

# Setup
- coming soon

# Evaluation
### Running Clio Batch

### Running Clio Online

### Places Evaluation

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
An overview of Clio is available on [YouTube](https://www.youtube.com/watch?v=m-HJO10qhSQ):