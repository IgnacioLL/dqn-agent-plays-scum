import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
from constants import Constants as C

def convert_to_binary_tensor(data: list[list[int]], pass_option: bool = False) -> torch.tensor:
    result = [[0] * 14 for _ in range(4)]

    # Convert each subarray (after the first) to a set for O(1) membership checks
    subarray_sets = [set(subarray) for subarray in data]

    # Iterate over each element in the first subarray
    for j in range(4):
        for i in range(1,15):
            # Check in each subsequent set
            if i in subarray_sets[j]:
                result[j][i-1] = 1
    if pass_option:
        return torch.tensor(np.append(np.array(result).flatten(), 1)).float().to(C.DEVICE)
    else:
        return torch.tensor(np.append(np.array(result).flatten(), 0)).float().to(C.DEVICE)

def print_rl_variables(reward: int, new_observation: np.array, finish: bool, epsilon: float) -> None:
        print(f"The reward is: {reward}")
        print(f"The new observation is: {new_observation}")
        print(f"The finish is: {finish}")
        print(f"The epsilon is: {epsilon}")

if __name__ == "__main__":
    test = [[1,2,3,4,5,6,7,8,9, 9, 9,11,11], [9,11], [9], []]
    binary_result = convert_to_binary(test)
    print(binary_result)