from sre_parse import BRANCH
from loguru import logger
import os
import sys
from pathlib import Path
import json
from channels_definition import *
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from edp_utils import get_raw_data_meta, get_raw_data_by_bag_name
import numpy as np
from pyquaternion import Quaternion as pyQuaternion
from scipy.interpolate import interp1d
import shutil
import tarfile
import time
from merge_lerobot_dataset import merge_datasets
import multiprocessing
import cv2
from cv_bridge import CvBridge
from ros_bag_extract.util import extract
from episode_parser.util import create_episode
from feature_spec.feature import create_features

# default: ROS 2
USE_ROS1 = bool(int(os.getenv('USE_ROS1', 0)))
# default: mp4 video
SAVE_VIDEO = bool(int(os.getenv('SAVE_VIDEO', 1)))
# default: AV1 codec
USE_H264 = bool(int(os.getenv('USE_H264', 0)))
# default: original file
USE_COMPRESSION = bool(int(os.getenv('USE_COMPRESSION', 1)))
# default: compute image stats
IS_COMPUTE_EPISODE_STATS_IMAGE = bool(int(os.getenv('IS_COMPUTE_EPISODE_STATS_IMAGE', 1)))
# default: 4 processes
MAX_PROCESSES = int(os.getenv('MAX_PROCESSES', 6))
# default: original description (possibly Chinese)
USE_TRANSLATION = bool(int(os.getenv('USE_TRANSLATION', 0)))

logger.info("env variables")
logger.info(f"USE_ROS1: {USE_ROS1}")
logger.info(f"SAVE_VIDEO: {SAVE_VIDEO}")
logger.info(f"USE_H264: {USE_H264}")
logger.info(f"USE_COMPRESSION: {USE_COMPRESSION}")
logger.info(f"IS_COMPUTE_EPISODE_STATS_IMAGE: {IS_COMPUTE_EPISODE_STATS_IMAGE}")
logger.info(f"MAX_PROCESSES: {MAX_PROCESSES}")
logger.info(f"USE_TRANSLATION: {USE_TRANSLATION}")

# Configure Loguru to avoid duplicate logs
# Remove the default sink before adding our own, otherwise messages
# may appear twice (default stderr sink + added stderr sink).
try:
    logger.remove()
    logger.add(sys.stderr, enqueue=True)
except Exception:
    pass

class DataConverter:
    def __init__(
            self, 
            robot_type, 
            sample_mcap_path, 
            dataset_name, 
            output_dir,
            cache_dir,
            use_ros1 = False, 
            save_video = False, 
            use_h264 = False, 
            use_compression = False, 
            is_compute_episode_stats_image = True,
            max_processes = 4,
            use_translation = False,
        ):

        self.dataset_name = dataset_name
        self.output_dir = os.path.join(output_dir, self.dataset_name)
        self.cache_dir = cache_dir
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        
        self.robot_type = robot_type
        logger.info(f'Robot type is: {self.robot_type}')
        self.shape_of_images = dict()
        
        self.use_ros1 = use_ros1
        self.save_video = save_video
        self.use_h264 = use_h264
        self.use_compression = use_compression
        self.is_compute_episode_stats_image = is_compute_episode_stats_image
        self.max_processes = max_processes
        self.use_translation = use_translation

        self.fps_dict = {}

        if not self.use_ros1:
            # check if wrist camera topic has "_rect", can be cleaned by ATC 2.1.5
            reader = SequentialReader()
            storage_options = StorageOptions(uri=sample_mcap_path, storage_id="mcap")
            converter_options = ConverterOptions()
            reader.open(storage_options, converter_options)
            all_topics = [topic.name for topic in reader.get_all_topics_and_types()]
            self.RGB_WRIST_LEFT_TOPIC = RGB_WRIST_LEFT_TOPIC
            self.RGB_WRIST_RIGHT_TOPIC = RGB_WRIST_RIGHT_TOPIC
            if self.RGB_WRIST_LEFT_TOPIC not in all_topics or self.RGB_WRIST_RIGHT_TOPIC not in all_topics:
                self.RGB_WRIST_LEFT_TOPIC = self.RGB_WRIST_LEFT_TOPIC.replace("image_raw", "image_rect_raw")
                self.RGB_WRIST_RIGHT_TOPIC = self.RGB_WRIST_RIGHT_TOPIC.replace("image_raw", "image_rect_raw")
            assert self.RGB_WRIST_LEFT_TOPIC in all_topics and self.RGB_WRIST_RIGHT_TOPIC in all_topics
        else:
            self.RGB_WRIST_LEFT_TOPIC = RGB_WRIST_LEFT_TOPIC
            self.RGB_WRIST_RIGHT_TOPIC = RGB_WRIST_RIGHT_TOPIC

        self.RGB_TOPICS = [
            self.RGB_WRIST_LEFT_TOPIC, 
            self.RGB_WRIST_RIGHT_TOPIC, 
            RGB_HEAD_RIGHT_TOPIC
        ]
        self.DEPTH_TOPICS = [
            DEPTH_HEAD_TOPIC, 
            DEPTH_LEFT_TOPIC, 
            DEPTH_RIGHT_TOPIC
        ]
        self.JOINT_TOPICS = [
            JOINT_OBS_LEFT_TOPIC, 
            JOINT_OBS_RIGHT_TOPIC, 
            GRIPPER_OBS_LEFT_TOPIC, 
            GRIPPER_OBS_RIGHT_TOPIC, 
            CHASSIS_OBS_TOPIC, 
            TORSO_OBS_TOPIC, 
            JOINT_ACTION_LEFT_TOPIC, 
            JOINT_ACTION_RIGHT_TOPIC, 
            TORSO_ACTION_TOPIC
        ]
        self.POSE_TOPICS = [
            EE_POSE_OBS_LEFT_TOPIC, 
            EE_POSE_OBS_RIGHT_TOPIC, 
        ]
        if self.robot_type == "r1pro":
            self.POSE_TOPICS.append(EE_POSE_ACTION_LEFT_TOPIC)
            self.POSE_TOPICS.append(EE_POSE_ACTION_RIGHT_TOPIC)
        self.GRIPPER_TOPICS = [
            GRIPPER_ACTION_LEFT_TOPIC, 
            GRIPPER_ACTION_RIGHT_TOPIC
        ]
        self.TWIST_TOPICS = [CHASSIS_ACTION_TOPIC]
        if self.robot_type == "r1lite":
            self.TWIST_TOPICS.append(TORSO_ACTION_SPEED_TOPIC)
        self.CONTROL_TOPICS = [
            JOINT_CONTROL_ACTION_LEFT_TOPIC, 
            JOINT_CONTROL_ACTION_RIGHT_TOPIC, 
            GRIPPER_CONTROL_ACTION_LEFT_TOPIC, 
            GRIPPER_CONTROL_ACTION_RIGHT_TOPIC, 
            CHASSIS_CONTROL_ACTION_TOPIC, 
            TORSO_CONTROL_ACTION_TOPIC
        ]
        self.IMU_TOPICS = [CHASSIS_IMU_TOPIC]
        self.TARGET_TOPICS = {
            GRIPPER_ACTION_LEFT_TOPIC: [],
            GRIPPER_ACTION_RIGHT_TOPIC: [],
            EE_POSE_OBS_LEFT_TOPIC: [],
            EE_POSE_OBS_RIGHT_TOPIC: [],
            JOINT_OBS_LEFT_TOPIC: [],
            JOINT_OBS_RIGHT_TOPIC: [],
            JOINT_ACTION_LEFT_TOPIC: [],
            JOINT_ACTION_RIGHT_TOPIC: [],
            GRIPPER_OBS_LEFT_TOPIC: [],
            GRIPPER_OBS_RIGHT_TOPIC: [],
            RGB_HEAD_LEFT_TOPIC: [],
            RGB_HEAD_RIGHT_TOPIC: [],
            self.RGB_WRIST_LEFT_TOPIC: [],
            self.RGB_WRIST_RIGHT_TOPIC: [],
            DEPTH_HEAD_TOPIC: [],
            DEPTH_LEFT_TOPIC: [],
            DEPTH_RIGHT_TOPIC: [],
            CHASSIS_ACTION_TOPIC: [],
            TORSO_ACTION_TOPIC: [],
            CHASSIS_OBS_TOPIC: [],
            CHASSIS_IMU_TOPIC: [],
            TORSO_OBS_TOPIC: [],
            JOINT_CONTROL_ACTION_LEFT_TOPIC: [],
            JOINT_CONTROL_ACTION_RIGHT_TOPIC: [],
            GRIPPER_CONTROL_ACTION_LEFT_TOPIC: [],
            GRIPPER_CONTROL_ACTION_RIGHT_TOPIC: [],
            CHASSIS_CONTROL_ACTION_TOPIC: [],
            TORSO_CONTROL_ACTION_TOPIC: []
        }
        if self.robot_type == "r1pro":
            self.TARGET_TOPICS[EE_POSE_ACTION_LEFT_TOPIC] = []
            self.TARGET_TOPICS[EE_POSE_ACTION_RIGHT_TOPIC] = []
        if self.robot_type == "r1lite":
            self.TARGET_TOPICS[TORSO_ACTION_SPEED_TOPIC] = []
        if self.robot_type == "r1pro":
            self.arm_dof = 7
        else:
            self.arm_dof = 6
    
    def msg_to_timestamp(self, msg):
        if not self.use_ros1:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        else:
            timestamp = msg.header.stamp.to_sec()
        return timestamp


    def process_all(self, mcaps_dict: dict):
        start_time = time.time()
        mcaps_dict = mcaps_dict["rawDataList"]
        total_files = len(mcaps_dict)
        logger.info("processing num: ", self.max_processes)
        args_list = [(idx, mcap_file) for idx, mcap_file in enumerate(mcaps_dict)]
        # self.process(0, mcaps_dict[0])
        if self.max_processes <= 1:
            for args in args_list:
                self.process_wrapper(*args)
        else:
            with multiprocessing.Pool(processes=self.max_processes, maxtasksperchild=1) as pool:
                pool.starmap(self.process_wrapper, args_list)
        
        process_time = time.time() - start_time
        
        start_time = time.time()
        self.merge_subdataset()
        merge_time = time.time() - start_time
        logger.info(f"Processing time: {process_time:.2f}s")
        logger.info(f"Merging time: {merge_time:.2f}s")

    def process_wrapper(self, idx, mcap_file):
        logger.info(f'Started processing {idx+1}: {mcap_file}')
        test_count = 0
        while True:
            self.process(idx, mcap_file)
            logger.info(f'Completed processing {idx+1}: {mcap_file}')
            break
        
    def compress_lerobot_ds(self,):
        if self.use_compression:
            def add_to_tar(tar, file_path):
                tar.add(file_path, arcname=os.path.basename(file_path))
            files = [os.path.join(self.cache_dir, self.dataset_name), 
                    os.path.join(self.cache_dir, "training_data_set_meta.json")]
            output_tar = os.path.join(self.cache_dir, f"{self.dataset_name}.tar.gz")
            with tarfile.open(output_tar, "w:gz") as tar:
                for file in files:
                    try:
                        add_to_tar(tar, file)
                    except Exception as e:
                        print(f"Error adding {file} to tar: {e}")
            shutil.move(output_tar, self.output_dir)

    def process(
            self, 
            idx, # for lerobot dataset merging
            mcap_info, # an element of rawDataList
        ):
        start_time = time.time()
        mcap_path = mcap_info["path"]
        processed_msgs = extract(mcap_path, self.use_ros1, self.TARGET_TOPICS)
        head_rgb_timestamps = np.array([self.msg_to_timestamp(msg) for msg in processed_msgs[RGB_HEAD_LEFT_TOPIC]])
        fps = int(np.round(1.0 / np.median(head_rgb_timestamps[1:] - head_rgb_timestamps[:-1])))
        self.fps_dict[str(idx)] = fps
        index_array = np.array([0, len(head_rgb_timestamps)])

        for topic, data in processed_msgs.items():
            if data is None:
                logger.warning(f'Message from {topic} is None!')
                continue
            
            if topic == RGB_HEAD_LEFT_TOPIC:
                bridge = CvBridge()
                head_rgb_images_list = []
                head_images = np.array([bridge.compressed_imgmsg_to_cv2(msg) for msg in data])
                self.shape_of_images["HEAD_LEFT_RGB"] = head_images[0].shape
                for i in range(len(index_array) - 1):
                    head_rgb_images_list.append(head_images[index_array[i]:index_array[i+1]])
                processed_msgs[topic] = head_rgb_images_list

            elif topic in self.RGB_TOPICS:
                bridge = CvBridge()
                wrist_rgb_images_list = []
                first_image_shape = bridge.compressed_imgmsg_to_cv2(data[0]).shape
                if topic == RGB_WRIST_LEFT_TOPIC:
                    self.shape_of_images["WRIST_LEFT_RGB"] = first_image_shape
                if topic == RGB_WRIST_RIGHT_TOPIC:
                    self.shape_of_images["WRIST_RIGHT_RGB"] = first_image_shape
                aligned_wrist_rgb_images = self.align_rgb(head_rgb_timestamps, data)
                for i in range(len(index_array) - 1):
                    wrist_rgb_images_list.append(aligned_wrist_rgb_images[index_array[i]:index_array[i+1]])
                processed_msgs[topic] = wrist_rgb_images_list
                
            elif topic in self.JOINT_TOPICS:
                timestamps = []
                positions = []
                velocities = []
                for msg in data:
                    timestamp = self.msg_to_timestamp(msg)
                    timestamps.append(timestamp)
                    positions.append(list(msg.position))
                    velocities.append(list(msg.velocity))
                interpolated_positions = self.interpolate_1d(head_rgb_timestamps, timestamps, positions)
                try:
                    interpolated_velocities = self.interpolate_1d(head_rgb_timestamps, timestamps, velocities)
                except Exception as e:
                    logger.error(f"Error interpolating velocities for mcap_path={mcap_path}: {e}")
                    raise
                joint_dict_list = list()
                for i in range(len(index_array) - 1):
                    joint_dict = dict()
                    joint_dict['position'] = interpolated_positions[index_array[i]:index_array[i+1]]
                    joint_dict['velocity'] = interpolated_velocities[index_array[i]:index_array[i+1]]
                    joint_dict_list.append(joint_dict)
                processed_msgs[topic] = joint_dict_list
            
            elif topic in self.POSE_TOPICS:
                pose_transforms = []
                pose_timestamps = []
                for msg in data:
                    timestamp = self.msg_to_timestamp(msg)
                    transform_ref = msg.pose
                    pos = [transform_ref.position.x, transform_ref.position.y, transform_ref.position.z]
                    quat = [transform_ref.orientation.x, transform_ref.orientation.y, transform_ref.orientation.z, transform_ref.orientation.w]
                    pose_transforms.append(pos + quat)
                    pose_timestamps.append(timestamp)
                pose_transforms = np.array(pose_transforms)
                pose_timestamps = np.array(pose_timestamps)
                transforms = self.interpolate_transform(head_rgb_timestamps, pose_timestamps, pose_transforms)

                pose_list = []
                for i in range(len(index_array) - 1):
                    pose_list.append(transforms[index_array[i]:index_array[i+1]])
                
                processed_msgs[topic] = pose_list

            elif topic in self.GRIPPER_TOPICS:
                timestamps = []
                positions = []
                gripper_list = []
                for msg in data:
                    timestamp = self.msg_to_timestamp(msg)
                    timestamps.append(timestamp)
                    positions.append([msg.position[0]])
                timestamps = np.array(timestamps)
                positions = np.array(positions)
                if len(positions) > 0:
                    positions = self.interpolate_1d(head_rgb_timestamps, timestamps, positions)
                
                for i in range(len(index_array) - 1):
                    gripper_list.append(positions[index_array[i]:index_array[i+1]])                
                processed_msgs[topic] = gripper_list

            elif topic in self.TWIST_TOPICS:
                timestamps = []
                velocities = []
                for msg in data:
                    timestamp = self.msg_to_timestamp(msg)
                    timestamps.append(timestamp)
                    velocities.append([
                        msg.twist.linear.x, 
                        msg.twist.linear.y, 
                        msg.twist.linear.z, 
                        msg.twist.angular.x, 
                        msg.twist.angular.y, 
                        msg.twist.angular.z
                    ])

                timestamps = np.array(timestamps)  # do not set dtype for timestamps, it exceeds the upper bound of fp32
                velocities = np.array(velocities)
                if len(velocities) > 0:
                    velocities = self.interpolate_1d(head_rgb_timestamps, timestamps, velocities)
                
                velocity_list = []
                for i in range(len(index_array) - 1):
                    velocity_list.append(velocities[index_array[i]:index_array[i+1]])
                
                processed_msgs[topic] = velocity_list
            
            elif topic in self.IMU_TOPICS:
                timestamps = []
                imu = []
                imu_transforms = []
                pos = [0.0, 0.0, 0.0]
                for msg in data:
                    timestamp = self.msg_to_timestamp(msg)
                    timestamps.append(timestamp)
                    imu.append([
                        msg.angular_velocity.x, 
                        msg.angular_velocity.y, 
                        msg.angular_velocity.z, 
                        msg.linear_acceleration.x, 
                        msg.linear_acceleration.y, 
                        msg.linear_acceleration.z
                    ])
                    imu_transforms.append(pos + [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

                timestamps = np.array(timestamps)
                imu = np.array(imu)
                imu_transforms = np.array(imu_transforms)
                imu = self.interpolate_1d(head_rgb_timestamps, timestamps, imu)
                transforms = self.interpolate_transform(head_rgb_timestamps, timestamps, imu_transforms)
                
                imu_list = []
                for i in range(len(index_array) - 1):
                    imu_list.append(np.concatenate((transforms[index_array[i]:index_array[i+1], 3:], imu[index_array[i]:index_array[i+1]]), axis=-1))
                
                processed_msgs[topic] = imu_list

            elif topic in self.CONTROL_TOPICS:
                timestamps = []
                positions = []
                velocities = []

                for msg in data:
                    timestamp = self.msg_to_timestamp(msg)
                    timestamps.append(timestamp)
                    positions.append(list(msg.p_des))
                    velocities.append(list(msg.v_des))

                if len(positions) > 0:
                    positions = self.interpolate_1d(head_rgb_timestamps, timestamps, positions)
                
                if len(velocities) > 0:
                    velocities = self.interpolate_1d(head_rgb_timestamps, timestamps, velocities)

                joint_dict_list = list()
                for i in range(len(index_array) - 1):
                    joint_dict = dict()
                    joint_dict['position'] = positions[index_array[i]:index_array[i+1]]
                    joint_dict['velocity'] = velocities[index_array[i]:index_array[i+1]]
                    joint_dict_list.append(joint_dict)
                processed_msgs[topic] = joint_dict_list

        episode = create_episode(processed_msgs, self.RGB_WRIST_LEFT_TOPIC, self.RGB_WRIST_RIGHT_TOPIC, self.arm_dof, self.robot_type)
        features = create_features(episode[0], self.arm_dof, self.save_video, self.shape_of_images, self.robot_type)
        lerobot_dataset = LeRobotDataset.create(
            repo_id=f'Galaxea/{self.dataset_name}_{idx}',
            features=features,
            robot_type=self.robot_type,
            root=os.path.join(self.cache_dir, self.dataset_name) + f'_{idx}',
            fps=fps,
        )

        # get coarse description and quality label from dcTask and labelsStr
        # as qualitySubLabel is used by base quality check
        episode_description = format_shelf_string(self.dataset_name)

        # get fine description and quality label from annotation.text and annotation.actionQualityLabel
        framewise_descriptions = []
        framewise_quality = []
        annotations = mcap_info.get("annotations", None)
        if self.use_translation:
            for annotation in annotations:
                annotation["translated_text"] = self.deepseek_translate_instruction(annotation["text"])
        for timestamp in head_rgb_timestamps:
            annotated = False
            # HACK: due to some mismatching between header timestamp and EDP, 
            # some early frames are not included in annotation interval
            if annotations is None:
                annotations = [dict(startSecond=0, startNanoSecond=0, endSecond=0, endNanoSecond=0)]
            else:
                annotations[0]["startSecond"] = 0
                annotations[0]["startNanoSecond"] = 0
            for cur_annotation in annotations:
                start_timestamp = cur_annotation["startSecond"] + cur_annotation["startNanoSecond"] / 1e9
                end_timestamp = cur_annotation["endSecond"] + cur_annotation["endNanoSecond"] / 1e9
                if timestamp > start_timestamp and timestamp < end_timestamp:
                    if self.use_translation:
                        description = f"{cur_annotation['text']}@{cur_annotation['translated_text']}"
                    else:
                        description = cur_annotation["text"]
                    framewise_descriptions.append(description)
                    framewise_quality.append("qualified" if cur_annotation["actionQualityLabel"] in ["None", "qualified"] else "unqualified")
                    annotated = True
                    break
            if not annotated:
                framewise_descriptions.append("null")
                framewise_quality.append("qualified")
        
        if len(episode) != 0:
            time_start = time.time()
            for frame, description, quality in zip(episode, framewise_descriptions, framewise_quality):
                lerobot_dataset.add_frame(
                    frame=frame,
                    task=episode_description#[episode_description, description, quality],
                )
            time_end = time.time()
            logger.info(f"add_frame time: {time_end - time_start} seconds")
            time_start = time.time()
            lerobot_dataset.save_episode()
            time_end = time.time()
            logger.info(f"save_episode time: {time_end - time_start} seconds")
            return (mcap_path, "success")
        else:
            return (mcap_path, "failed")

    
    
    def merge_subdataset(self,):
        subdatasets = []
        for entry in os.scandir(self.cache_dir):
            if entry.is_dir():
                full_path = entry.path
                subdatasets.append(full_path)
        subdatasets = sorted(subdatasets)
        logger.info(f'Start merging:\n {subdatasets}\n To: {self.cache_dir}/{self.dataset_name}')

        sum_fps = 0
        count = 0
        for key, fps in self.fps_dict.items():
            sum_fps += fps
            count += 1
        mean_fps = sum_fps // count if count > 0 else 15

        merge_datasets(
            source_folders=subdatasets,
            output_folder=os.path.join(self.cache_dir, self.dataset_name),
            default_fps=mean_fps, 
        )
    
    def align_rgb(self, head_rgb_timestamps, source_data):
        source_timestamps = []
        source_images = []
        bridge = CvBridge()
        for msg in source_data:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            source_timestamps.append(timestamp)
            source_images.append(bridge.compressed_imgmsg_to_cv2(msg))
        
        source_timestamps = np.array(source_timestamps)
        source_images = np.array(source_images)
        
        closest_indices = np.abs(source_timestamps[:, None] - head_rgb_timestamps).argmin(axis=0)
        source_images = np.array(source_images)
        aligned_images = source_images[closest_indices]

        return aligned_images

    def interpolate_1d(self, target_timestamps, source_timestamps, source_values):
        noise_topic_threshold = 0.2
        if len(source_timestamps) < noise_topic_threshold * len(target_timestamps):
            return source_values
        
        f = interp1d(source_timestamps, source_values, kind='linear', axis=0, bounds_error=False, fill_value=(source_values[0], source_values[-1]))
        interpolated_states = f(target_timestamps)

        return interpolated_states
        
    def interpolate_transform(self, target_timestamps, source_timestamps, source_values):
        positions = source_values[:, :3]
        quaternions = [pyQuaternion(source_values[i, 6], source_values[i, 3], source_values[i, 4], source_values[i, 5]) for i in range(source_values.shape[0])]

        target_values = np.empty((len(target_timestamps), 7))

        # Use numpy's searchsorted to find indices for interpolation
        indices = np.searchsorted(source_timestamps, target_timestamps, side='right')

        # Ensure indices are within valid range
        indices = np.clip(indices, 1, len(source_timestamps) - 1)

        t0 = source_timestamps[indices - 1]
        t1 = source_timestamps[indices]

        pos0 = positions[indices - 1]
        pos1 = positions[indices]

        quat0 = np.array([quaternions[i - 1] for i in indices])
        quat1 = np.array([quaternions[i] for i in indices])

        # Calculate interpolation factors
        t_interp = (target_timestamps - t0) / (t1 - t0)

        target_values[:, :3] = pos0 + np.expand_dims(t_interp, axis=1) * (pos1 - pos0)
        
        interpolated_quats = [pyQuaternion.slerp(quat0[i], quat1[i], t_interp[i]).normalised for i in range(len(t_interp))]
        target_values[:, 3:] = np.array([[q.x, q.y, q.z, q.w] for q in interpolated_quats])

        # For timestamps out of the source range
        target_values[target_timestamps <= source_timestamps[0], :3] = positions[0]
        target_values[target_timestamps <= source_timestamps[0], 3:] = [quaternions[0].x, quaternions[0].y, quaternions[0].z, quaternions[0].w]
        
        target_values[target_timestamps >= source_timestamps[-1], :3] = positions[-1]
        target_values[target_timestamps >= source_timestamps[-1], 3:] = [quaternions[-1].x, quaternions[-1].y, quaternions[-1].z, quaternions[-1].w]

        return target_values
    
    def register_quat(self, a):
        a_np = np.array(a)
        quat = a_np[:, 3: 7]
        
        quat_couple_number = 2
        dot = np.diag(np.dot(quat[0: -1], quat[1:].T))
        
        shift_indices = np.where(dot < 0)[0] + 1
        if len(shift_indices) % quat_couple_number == 1:
            shift_indices = np.append(shift_indices, len(quat))
        fixed_quat = quat.copy()
        for i in range(len(shift_indices) // quat_couple_number):
            fixed_quat[shift_indices[2 * i]: shift_indices[2 * i + 1]] = -fixed_quat[shift_indices[2 * i]: shift_indices[2 * i + 1]]
        assert fixed_quat.shape == quat.shape
        a_np[:, 3: 7] = fixed_quat
        a[...] = a_np

    def deepseek_translate_instruction(self, text, temperature=0.0, max_tokens=200):
        # from openai import OpenAI
        from volcenginesdkarkruntime import Ark

        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        client = Ark(api_key=deepseek_api_key)

        cnt = 0
        while cnt < 5:
            try:
                response = client.chat.completions.create(
                    model="deepseek-v3-1-250821",
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的机器人指令翻译专家。"
                        },
                        {
                            "role": "user",
                            "content": f"请将以下中文机器人动作指令翻译成准确简洁、明确的英文，使用祈使句格式，不要添加任何解释：\n\n{text}"
                        }
                    ],
                    thinking={
                        "type": "disabled"
                    },
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = response.choices[0].message.content
                text = text.replace("\n", " ")
                text = " ".join(text.split())
                return text
            except Exception as e:
                cnt += 1
                print(f'Failed to translate, {e}, retry {cnt}')
                continue
        return None

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='ROS to LeRobot data')
    parser.add_argument('--input_dir', required=True, type=str, help='where ROS  are saved')
    parser.add_argument('--output_dir', required=True, type=str, help='where LeRobot data are saved')
    parser.add_argument('--robot_type', required=True, choices=['R1Pro', 'R1', 'R1Lite'], help='robot type')
    parser.add_argument('--dataset_name', required=True, type=str, help='dataset_name')

    return parser.parse_args()

def search_rosbags(input_dir: str):
    path = Path(input_dir)
    try:
        bag_files = list(path.rglob('*.bag'))
    except Exception as e:
        print(f"Error searching for bag files: {e}")
    if len(bag_files) == 0:
        bag_files = list(path.rglob('*.mcap'))
    return sorted([str(file.absolute()) for file in bag_files])

def get_raw_data_meta_from_args():
    args = get_args()
    dataset_name = args.dataset_name
    robot_type = args.robot_type
    output_dir = args.output_dir
    cache_dir = f"/tmp/{args.dataset_name}/"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)

    data_path_list = search_rosbags(args.input_dir)

    raw_data_meta_path = os.path.join(args.input_dir, "raw_data_meta.json")
    if os.path.exists(raw_data_meta_path):
        with open(raw_data_meta_path) as f:
            raw_data_meta_json = json.load(f)
            raw_data_meta_json = dict(data=raw_data_meta_json)
            raw_data_meta_json["data"]["rawDataSetName"] = dataset_name
    else:
        raw_data_meta_dict = dict()
        raw_data_meta_dict["rawDataSetName"] = dataset_name
        raw_data_meta_dict["rawDataList"] = list()
        for data_path in data_path_list:
            data_meta = dict()
            data_meta['name'] = os.path.basename(data_path)
            data_meta['path'] = data_path
            data_meta['robotType'] = robot_type
            raw_data_meta_dict["rawDataList"].append(data_meta)

        raw_data_meta_json = dict(data=raw_data_meta_dict)
    return (raw_data_meta_json, cache_dir, output_dir)

def format_shelf_string(s):
    parts = s.split('_')
    filtered_parts = [part for part in parts if not part.isdigit()]
    final_parts = [part for part in filtered_parts if not part.isdigit() or len(part) < 8]
    result = ' '.join(final_parts).lower()
    return result

if __name__ == '__main__':
    if os.path.exists("/edp-workspace/instance-env/"):
        raw_data_meta_json, cache_dir, output_dir = get_raw_data_meta()
    else:
        raw_data_meta_json, cache_dir, output_dir = get_raw_data_meta_from_args()

    mcaps_dict = raw_data_meta_json['data']
    dataset_name = mcaps_dict['rawDataSetName']
    robot_type = mcaps_dict['rawDataList'][0]['robotType'].lower()
    sample_mcap_path = mcaps_dict["rawDataList"][0]["path"]
    bag_type = mcaps_dict['rawDataList'][0]['name'].split('.')[-1]

    if bag_type == 'bag':
        USE_ROS1 = True
    else:
        USE_ROS1 = False

    if not USE_ROS1:
        import rclpy
        from rclpy.serialization import deserialize_message
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        logger.info(f"Use ROS1")
    else:
        import rosbag
        logger.info(f"Use ROS2")

    if not USE_ROS1:
        rclpy.init()

    if robot_type not in ['r1pro', 'r1', 'r1lite']:
        logger.error(f'Unknown robot type: {robot_type}')
        exit(1)

    try:
        training_data_set_meta_file = os.path.join(cache_dir, 'training_data_set_meta.json')
        with open(training_data_set_meta_file, "w", encoding="utf-8") as f:
            json.dump(mcaps_dict, f, indent=4)
    except Exception as e:
        print("No you dont")
        logger.error(f'Error writing {training_data_set_meta_file}, {e}')

    data_converter = DataConverter(
        robot_type, 
        sample_mcap_path, 
        dataset_name, 
        output_dir,
        cache_dir,
        use_ros1=USE_ROS1, 
        save_video=SAVE_VIDEO, 
        use_h264=USE_H264, 
        use_compression=USE_COMPRESSION, 
        is_compute_episode_stats_image=IS_COMPUTE_EPISODE_STATS_IMAGE,
        max_processes=MAX_PROCESSES,
        use_translation=USE_TRANSLATION
    )

    # 2. process messages.
    data_converter.process_all(mcaps_dict)

    data_converter.compress_lerobot_ds()
    
    if not USE_ROS1:    
        rclpy.shutdown()
