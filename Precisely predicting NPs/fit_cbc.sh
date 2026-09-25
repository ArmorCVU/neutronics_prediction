#!/bin/bash

all_modules=("res" "res_se" "inception" "mix")

# 遍历数组并打印每个元素
for item in "${all_modules[@]}"; do
    python fit_cbc.py --module $item --device cuda:0
done

