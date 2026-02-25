# GalaxeaLeRobotToolkit
Convert rosbag / mcap to lerobot

# For updated 1.1 Version
  - If you run the script locally, temporary lerobot datasets before merging will be stored at /tmp

## Enviroment Requirements
1. source Ros2 Humble setup.bash
2. conda env with python3.10
3. pip install dependencies below:
```bash
pip install git+https://github.com/OpenGalaxea/GalaxeaLeRobot.git
pip install loguru pyquaternion scipy
```
4. build hdas_msg
```bash
colcon build --packages-select hdas_msg
```

## Usage
Edit dataset_name, input_dir, output_dir and robot_type in run.sh, then
```bash
# build hdas_msg before source
source install/setup.bash 
bash run.sh
```
