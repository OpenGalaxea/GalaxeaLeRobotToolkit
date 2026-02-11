import time
import os
from loguru import logger


def extract(bag_file, use_ros1, target_topics):
        if not use_ros1:
            return extract_ros2(bag_file, target_topics)
        else:
            return extract_ros1(bag_file, target_topics)

def extract_ros1(bag_file, target_topics):
        import rosbag
        time_start = time.time()
        extracted_msgs = {topic : [] for topic in target_topics}
        bag = rosbag.Bag(bag_file)
        for topic, msg, t in bag.read_messages():
            if topic in extracted_msgs.keys():
                extracted_msgs[topic].append(msg)
        bag.close()
        time_end = time.time()
        logger.info(f"extract_ros1 time: {time_end - time_start} seconds")
        return extracted_msgs

def extract_ros2(mcap_file, target_topics):
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    import rclpy
    from rclpy.serialization import deserialize_message
    time_start = time.time()
    mcap_name = os.path.basename(mcap_file)
    # logger.info(f"Loading {mcap_name} mcap file.")
    extracted_msgs = {topic : [] for topic in target_topics}
    reader = SequentialReader()
    storage_options = StorageOptions(uri=mcap_file, storage_id="mcap")
    converter_options = ConverterOptions()
    reader.open(storage_options, converter_options)
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_type = type_map.get(topic)
            
        if not msg_type:
            logger.warning(f'Unknown topic type: {topic}')
            continue

        try:
            if topic in extracted_msgs.keys():
                module_name, class_name = msg_type.rsplit('/', 1)
                module = __import__(f'{module_name.replace("/", ".")}', fromlist=[class_name])
                msg_class = getattr(module, class_name)
                msg = deserialize_message(data, msg_class)
                # msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                extracted_msgs[topic].append(msg)
        except Exception as e:
            logger.error(f'Error processing {topic}: {str(e)}')
    time_end = time.time()
    logger.info(f"extract_ros2 time: {time_end - time_start} seconds")
    return extracted_msgs