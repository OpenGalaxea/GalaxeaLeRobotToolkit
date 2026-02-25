from channels_definition import *
import numpy as np

def create_episode(processed_dataset, RGB_WRIST_LEFT_TOPIC, RGB_WRIST_RIGHT_TOPIC, arm_dof, robot_type):
    episode = []
    for i in range(len(processed_dataset[RGB_HEAD_LEFT_TOPIC][0])):
        frame = {}
        frame["observation.images.head_rgb"] = processed_dataset[RGB_HEAD_LEFT_TOPIC][0][i]
        frame["observation.images.head_right_rgb"] = processed_dataset[RGB_HEAD_RIGHT_TOPIC][0][i]
        frame["observation.images.left_wrist_rgb"] = processed_dataset[RGB_WRIST_LEFT_TOPIC][0][i]
        frame["observation.images.right_wrist_rgb"] = processed_dataset[RGB_WRIST_RIGHT_TOPIC][0][i]
        
        frame["observation.state.left_arm"] = processed_dataset[JOINT_OBS_LEFT_TOPIC][0]["position"][i][0: arm_dof]
        frame["observation.state.left_arm.velocities"] = processed_dataset[JOINT_OBS_LEFT_TOPIC][0]["velocity"][i][0: arm_dof]
        frame["observation.state.right_arm"] = processed_dataset[JOINT_OBS_RIGHT_TOPIC][0]["position"][i][0: arm_dof]
        frame["observation.state.right_arm.velocities"] = processed_dataset[JOINT_OBS_RIGHT_TOPIC][0]["velocity"][i][0: arm_dof]
        frame["observation.state.left_gripper"] = processed_dataset[GRIPPER_OBS_LEFT_TOPIC][0]["position"][i]
        frame["observation.state.right_gripper"] = processed_dataset[GRIPPER_OBS_RIGHT_TOPIC][0]["position"][i]
        frame["observation.state.chassis.imu"] = processed_dataset[CHASSIS_IMU_TOPIC][0][i]
        frame["observation.state.chassis"] = processed_dataset[CHASSIS_OBS_TOPIC][0]['position'][i][0:3]
        # FIXME: The feedback for the chassis provides a 6-dim velocity, 
        # but only the first 3 dims are valid. The last 3 dims do not change.
        frame["observation.state.chassis.velocities"] = processed_dataset[CHASSIS_OBS_TOPIC][0]['velocity'][i][0:3]
        frame["observation.state.torso"] = processed_dataset[TORSO_OBS_TOPIC][0]["position"][i]
        frame["observation.state.torso.velocities"] = processed_dataset[TORSO_OBS_TOPIC][0]["velocity"][i]
        frame["observation.state.left_ee_pose"] = processed_dataset[EE_POSE_OBS_LEFT_TOPIC][0][i]
        frame["observation.state.right_ee_pose"] = processed_dataset[EE_POSE_OBS_RIGHT_TOPIC][0][i]
        
        if robot_type == "r1pro":
            frame["action.left_ee_pose"] = processed_dataset[EE_POSE_ACTION_LEFT_TOPIC][0][i]
            frame["action.right_ee_pose"] = processed_dataset[EE_POSE_ACTION_RIGHT_TOPIC][0][i]
        
        frame["action.left_gripper"] = processed_dataset[GRIPPER_ACTION_LEFT_TOPIC][0][i]
        frame["action.right_gripper"] = processed_dataset[GRIPPER_ACTION_RIGHT_TOPIC][0][i]
        frame["action.left_arm"] = processed_dataset[JOINT_ACTION_LEFT_TOPIC][0]["position"][i]
        frame["action.right_arm"] = processed_dataset[JOINT_ACTION_RIGHT_TOPIC][0]["position"][i]
        frame["action.chassis.velocities"] = processed_dataset[CHASSIS_ACTION_TOPIC][0][i]
        
        # only R1 Pro with whole-body control has torso joint action, while R1 Lite still uses torso speed control
        if robot_type == "r1pro" and len(processed_dataset[TORSO_ACTION_TOPIC][0]["position"]) > 0:
            frame["action.torso"] = processed_dataset[TORSO_ACTION_TOPIC][0]["position"][i]
        if robot_type == "r1lite" and len(processed_dataset[TORSO_ACTION_SPEED_TOPIC][0]) > 0:
            frame["action.torso.velocities"] = processed_dataset[TORSO_ACTION_SPEED_TOPIC][0][i]

        episode.append(frame)
    
    return episode