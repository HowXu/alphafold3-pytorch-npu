#!/bin/bash

# 设置刷新间隔（秒）
interval=5

echo "开始监控NPU运行状态，每${interval}秒刷新一次..."

while true
do
    clear
    current_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "============================================"
    echo "时间: ${current_time}"
    npu-smi info
    echo "============================================"
    sleep $interval
done