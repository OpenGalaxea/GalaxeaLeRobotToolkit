import os
import time
import hashlib
import requests
import json
from pathlib import Path
import subprocess
import random


def cal_auth(ak, sk):
    timestamp_seconds = time.time()
    timestamp_milliseconds = int(timestamp_seconds * 1e3)
    string_to_encrypt = sk + "," + str(timestamp_milliseconds)
    encoded_string = string_to_encrypt.encode('utf-8')
    sha256_hash = hashlib.sha256()
    sha256_hash.update(encoded_string)
    encrypted_string = sha256_hash.hexdigest()
    header = "Digest " + ak + ";" + str(timestamp_milliseconds) + ";" + encrypted_string
    
    return header

def request_with_retry(method, url, headers=None, data=None, json_body=None, timeout=60,
    max_retries=5, backoff_base_seconds=0.5, max_backoff_seconds=8.0,
    retry_on_status=(429, 500, 502, 503, 504)):
    """
    使用指数退避+随机抖动的重试机制封装 requests.request。

    参数:
        method: HTTP 方法，如 "GET"、"POST"。
        url: 请求地址。
        headers: 请求头。
        data: 表单或字节数据。
        json_body: JSON 负载（与 data 互斥）。
        timeout: 每次请求的超时时间（秒）。
        max_retries: 最大重试次数（不含首次）。
        backoff_base_seconds: 退避基数，实际退避为 base * 2^attempt，并加入抖动。
        max_backoff_seconds: 退避最大值上限。
        retry_on_status: 需要重试的 HTTP 状态码集合。

    返回:
        requests.Response 对象。
    """
    attempt_index = 0
    while True:
        try:
            ak = os.getenv('EDP_AK')
            sk = os.getenv('EDP_SK')
            cal_auth_value = cal_auth(ak, sk)
            headers['Authorization'] = cal_auth_value
            response = requests.request(
                method,
                url,
                headers=headers,
                data=data,
                json=json_body,
                timeout=timeout,
            )
            # 对特定 HTTP 状态码进行重试
            if response.status_code in retry_on_status and attempt_index < max_retries:
                # 计算指数退避时间，加入[0, backoff]的抖动
                backoff_no_jitter = min(
                    max_backoff_seconds,
                    backoff_base_seconds * (2 ** attempt_index)
                )
                sleep_seconds = random.uniform(0, backoff_no_jitter)
                print(f"request retryable status {response.status_code}, retrying in {sleep_seconds:.2f}s (attempt {attempt_index + 1}/{max_retries})")
                time.sleep(sleep_seconds)
                attempt_index += 1
                continue
            # 非重试状态码或已达到重试上限
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if attempt_index >= max_retries:
                # 达到重试上限，抛出异常
                raise
            backoff_no_jitter = min(
                max_backoff_seconds,
                backoff_base_seconds * (2 ** attempt_index)
            )
            sleep_seconds = random.uniform(0, backoff_no_jitter)
            print(f"request exception: {exc}, retrying in {sleep_seconds:.2f}s (attempt {attempt_index + 1}/{max_retries})")
            time.sleep(sleep_seconds)
            attempt_index += 1

def get_raw_data_meta():
    ak = os.getenv('EDP_AK')
    sk = os.getenv('EDP_SK')
    dataset_name = os.getenv('RAW_DATA_SET_NAME')
    version = os.getenv('VERSION')
    cache_dir = "/edp-workspace/instance-env/"
    url = f"http://edp-app-backend-service.edp:8081/edp-app-be/backend/v1/business/training-data-set/get-meta?rawDataSetName={dataset_name}&version={version}"
    payload = {}
    cal_auth_value = cal_auth(ak, sk)
    headers = {
        'accept': '*/*',
        'Authorization': cal_auth_value
    }
    response = request_with_retry("GET", url, headers=headers, data=payload)
    raw_data_meta_json = json.loads(response.text)
    output_dir = os.getenv('TRAINING_DATA_SET_DIR')
    return (raw_data_meta_json, cache_dir, output_dir)

def get_raw_data_by_bag_name(bag_name):
    ak = os.getenv('EDP_AK')
    sk = os.getenv('EDP_SK')
    url = "http://edp-app-backend-service.edp:8081/edp-app-be/backend/v1/business/raw-data/query"
    payload = json.dumps({
        "bagName": bag_name,
        "pageNum": 1,
        "pageSize": -1
    })
    headers = {
        'accept': '*/*',
        'Content-Type': 'application/json',
        'Authorization': cal_auth(ak, sk)
    }
    response = request_with_retry("POST", url, headers=headers, data=payload)
    data = json.loads(response.text)
    return data
# get_raw_data_by_bag_name("RB250417001_20250805214344892_RAW.mcap")
