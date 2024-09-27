### Installation

See rosinstall file for approximate branches. You will need to install `semantic_inference` by following the open-set instructions (in particular for the ros environment)

### Running
You may want to make a rosbag (see instructions in semantic_inference) instead of using the live segmentation.

```
source /path/to/semantic_inference/env/bin/activate
roslaunch clio_ros realsense.launch verbose_visualizer:=true place_tasks:="[office, cubicle, kitchen]", object_tasks:="[chair, cabinet, desk, printer]"
```
