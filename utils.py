import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def convert_to_binary(data: list[list[int]]) -> np.array:
    result = [[0] * 14 for _ in range(4)]

    # Convert each subarray (after the first) to a set for O(1) membership checks
    subarray_sets = [set(subarray) for subarray in data]

    # Iterate over each element in the first subarray
    for j in range(4):
        for i in range(1,15):
            # Check in each subsequent set
            if i in subarray_sets[j]:
                result[j][i-1] = 1
    return np.array(result).flatten()

if __name__ == "__main__":
    test = [[1,2,3,4,5,6,7,8,9, 9, 9,11,11], [9,11], [9], []]
    binary_result = convert_to_binary(test)
    print(binary_result)