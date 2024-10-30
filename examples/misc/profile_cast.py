#!/usr/bin/env python3

"""
This script is used to cast the profile data from Jolteon format to
the Flexecutor format
"""

import json

# --- ML
# config_pairs = [
#     [1024, 8],
#     [1024, 16],
#     [1024, 32],
#     [1792, 4],
#     [1792, 8],
#     [1792, 16],
#     [1792, 32],
#     [3584, 4],
#     [3584, 8],
#     [3584, 16],
#     [3584, 32],
#     [5120, 4],
#     [5120, 8],
#     [5120, 16],
#     [5120, 32],
#     [6144, 4],
#     [6144, 8],
#     [6144, 16],
#     [6144, 32],
#     [7168, 4],
#     [7168, 8],
#     [7168, 16],
#     [7168, 32],
# ]

# --- Video Analytics
config_pairs = [
    [1792, 4],
    [1792, 8],
    [1792, 16],
    [1792, 32],
    [3584, 4],
    [3584, 8],
    [3584, 16],
    [3584, 32],
    [5120, 4],
    [5120, 8],
    [5120, 16],
    [5120, 32],
    [6144, 4],
    [6144, 8],
    [6144, 16],
    [6144, 32],
    [7168, 4],
    [7168, 8],
    [7168, 16],
    [7168, 32],
    [8960, 4],
    [8960, 8],
    [8960, 16],
    [8960, 32],
]

profile_path = "/home/manri-urv/urv/Jolteon/profiles/Video-Analytics_profile.json"

stages = ["stage0", "stage1", "stage2", "stage3"]

for stage in stages:
    dict = {}
    with open(profile_path) as f:
        data = json.load(f)

    for index, config in enumerate(config_pairs):
        tuple_key = str((1, config[0], config[1]))
        dict[tuple_key] = {"read": [], "write": [], "cold_start": [], "compute": []}

        for i in range(len(data[stage]["cold"])):
            dict[tuple_key]["read"].append(data[stage]["read"][i][index])
            dict[tuple_key]["compute"].append(data[stage]["compute"][i][index])
            dict[tuple_key]["write"].append(data[stage]["write"][i][index])
            dict[tuple_key]["cold_start"].append(data[stage]["cold"][i][index])

    with open(f"{stage}.json", "w") as f:
        json.dump(dict, f)
