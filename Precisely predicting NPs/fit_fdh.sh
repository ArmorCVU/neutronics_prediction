#!/bin/bash

all_modules=("inception" "mix" "mix_se" "res" "res_se")

# 遍历数组并打印每个元素
for item in "${all_modules[@]}"; do
    python fit_fdh.py --module $item --device cuda:3
done