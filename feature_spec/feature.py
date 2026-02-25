from channels_definition import *
import pyarrow as pa

def create_features(frame_sample, arm_dof, save_video, shape_of_images, robot_type):
        features = {}
        # RGB
        image_dtype = "video" if save_video else "image"
        features["observation.images.head_rgb"] = {
            "dtype": image_dtype,
            "shape": shape_of_images["HEAD_LEFT_RGB"],
            "names": ["height", "width", "channels"],
        }

        features["observation.images.head_right_rgb"] = {
            "dtype": image_dtype,
            "shape": shape_of_images["HEAD_LEFT_RGB"],
            "names": ["height", "width", "channels"],
        }

        features["observation.images.left_wrist_rgb"] = {
            "dtype": image_dtype,
            "shape": shape_of_images["WRIST_LEFT_RGB"],
            "names": ["height", "width", "channels"],
        }

        features["observation.images.right_wrist_rgb"] = {
            "dtype": image_dtype,
            "shape": shape_of_images["WRIST_RIGHT_RGB"],
            "names": ["height", "width", "channels"],
        }

        # Arm Joints
        arm_feat = {
            "dtype": "float64",
            "shape": (arm_dof,),
            "names": None
        }
        features["observation.state.left_arm"] = arm_feat.copy()
        features["observation.state.left_arm"]["names"] = [JOINT_OBS_LEFT_TOPIC+f".position[{i}]" for i in range(arm_dof)]
        features["observation.state.left_arm.velocities"] = arm_feat.copy()
        features["observation.state.left_arm.velocities"]["names"] = [JOINT_OBS_LEFT_TOPIC+f".velocity[{i}]" for i in range(arm_dof)]
        features["observation.state.right_arm"] = arm_feat.copy()
        features["observation.state.right_arm"]["names"] = [JOINT_OBS_RIGHT_TOPIC+f".position[{i}]" for i in range(arm_dof)]
        features["observation.state.right_arm.velocities"] = arm_feat.copy()
        features["observation.state.right_arm.velocities"]["names"] = [JOINT_OBS_RIGHT_TOPIC+f".velocity[{i}]" for i in range(arm_dof)]

        imu_names = [
            ".orientation.x",
            ".orientation.y",
            ".orientation.z",
            ".orientation.w",
            ".angular_velocity.x",
            ".angular_velocity.y",
            ".angular_velocity.z",
            ".linear_acceleration.x",
            ".linear_acceleration.y",
            ".linear_acceleration.z"
        ]
        # Chassis
        features["observation.state.chassis.imu"] = {
            "dtype": "float64",
            "shape": (10,),
            "names": [CHASSIS_IMU_TOPIC+name for name in imu_names]
        }

        chassis_obs_names = [
            ".position[0]",
            ".position[1]",
            ".position[2]",
            ".velocity[0]",
            ".velocity[1]",
            ".velocity[2]",
        ]
        features["observation.state.chassis"] = {
            "dtype": "float64",
            "shape": (3,),
            "names": [CHASSIS_OBS_TOPIC+name for name in chassis_obs_names[:3]]
        }
        features["observation.state.chassis.velocities"] = {
            "dtype": "float64",
            "shape": (3,),
            "names": [CHASSIS_OBS_TOPIC+name for name in chassis_obs_names[3:]]
        }

        # Torso
        torso_obs_names = [
            ".position[0]",
            ".position[1]",
            ".position[2]",
            ".position[3]",
            ".velocity[0]",
            ".velocity[1]",
            ".velocity[2]",
            ".velocity[3]",
        ]
        features["observation.state.torso"] = {
            "dtype": "float64",
            "shape": (4,),
            "names": [TORSO_OBS_TOPIC+name for name in torso_obs_names[:4]]
        }

        features["observation.state.torso.velocities"] = {
            "dtype": "float64",
            "shape": (4,),
            "names": [TORSO_OBS_TOPIC+name for name in torso_obs_names[4:]]
        }

        # Gripper
        features["observation.state.left_gripper"] = {
            "dtype": "float64",
            "shape": (1,),
            "names": [GRIPPER_OBS_LEFT_TOPIC+".position[0]"]
        }

        features["observation.state.right_gripper"] = {
            "dtype": "float64",
            "shape": (1,),
            "names": [GRIPPER_OBS_RIGHT_TOPIC+".position[0]"]
        }

        # EE
        eef_pose = {
            "dtype": "float64",
            "shape": (7,),
            "names": None
        }
        pose_names = [
            ".pose.position.x",
            ".pose.position.y",
            ".pose.position.z",
            ".pose.orientation.x",
            ".pose.orientation.y",
            ".pose.orientation.z",
            ".pose.orientation.w"
        ]
        features["observation.state.left_ee_pose"] = eef_pose.copy()
        features["observation.state.left_ee_pose"]["names"] = [EE_POSE_OBS_LEFT_TOPIC+name for name in pose_names]
        features["observation.state.right_ee_pose"] = eef_pose.copy()
        features["observation.state.right_ee_pose"]["names"] = [EE_POSE_OBS_RIGHT_TOPIC+name for name in pose_names]

        # Actions
        if robot_type == "r1pro":
            features["action.left_ee_pose"] = eef_pose.copy()
            features["action.left_ee_pose"]["names"] = [EE_POSE_ACTION_LEFT_TOPIC+name for name in pose_names]
            features["action.right_ee_pose"] = eef_pose.copy()
            features["action.right_ee_pose"]["names"] = [EE_POSE_ACTION_RIGHT_TOPIC+name for name in pose_names]
        
        features["action.left_gripper"] = {
            "dtype": "float64",
            "shape": (1,),
            "names": [GRIPPER_ACTION_LEFT_TOPIC+".position[0]"]
        }
        features["action.right_gripper"] = {
            "dtype": "float64",
            "shape": (1,),
            "names": [GRIPPER_ACTION_RIGHT_TOPIC+".position[0]"]
        }

        chassis_twist_names = [
            ".twist.linear.x",
            ".twist.linear.y",
            ".twist.linear.z",
            ".twist.angular.x",
            ".twist.angular.y",
            ".twist.angular.z",
        ]
        features["action.chassis.velocities"] = {
            "dtype": "float64", 
            "shape": (6,), 
            "names": [CHASSIS_ACTION_TOPIC+name for name in chassis_twist_names]
        }

        # NOTE: torso will have two different control types, and
        # will not record both of them in the same episode.
        if "action.torso" in frame_sample:
            features["action.torso"] = {
                "dtype": "float64",
                "shape": (4,), 
                "names": [TORSO_ACTION_TOPIC+f".position[{i}]" for i in range(4)]
            }

        if "action.torso.velocities" in frame_sample:
            pose_names = [
                ".twist.linear.x",
                ".twist.linear.y",
                ".twist.linear.z",
                ".twist.angular.x",
                ".twist.angular.y",
                ".twist.angular.z",
            ]
            features["action.torso.velocities"] = {
                "dtype": "float64",
                "shape": (6,),
                "names": [TORSO_ACTION_SPEED_TOPIC+name for name in pose_names]
            }
        
        features["action.left_arm"] = {
                "dtype": "float64",
                "shape": (arm_dof,),
                "names": None
            }
        features["action.left_arm"]["names"] = [JOINT_ACTION_LEFT_TOPIC+f".position[{i}]" for i in range(arm_dof)]
        features["action.right_arm"] = {
                "dtype": "float64",
                "shape": (arm_dof,),
                "names": None
            }
        features["action.right_arm"]["names"] = [JOINT_ACTION_RIGHT_TOPIC+f".position[{i}]" for i in range(arm_dof)]
        
        return features