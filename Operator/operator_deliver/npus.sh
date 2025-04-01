#!/bin/bash

# 保存上一次 npu-smi 的输出
prev_output=""

# 无限循环，每隔 1 秒运行一次 npu-smi
while true; do
    # 获取当前 npu-smi 的输出
    current_output=$(npu-smi info)

    # 比较当前输出和上一次输出
    if [ "$current_output" != "$prev_output" ]; then
        # 如果有变化，清空屏幕并显示当前输出
        clear
        echo "$current_output"
        prev_output="$current_output"
    fi

    # 等待 1 秒
    sleep 1
done