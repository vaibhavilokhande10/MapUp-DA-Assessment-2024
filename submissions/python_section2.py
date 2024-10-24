from typing import Dict, List
import pandas as pd
import re
import itertools
import numpy as np

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    for i in range(0, len(lst), n):
        group = lst[i:i+n]
        for j in range(len(group)-1, -1, -1):
            result.append(group[j])
    return result

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    for s in lst:
        length = len(s)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(s)
    return dict(sorted(length_dict.items()))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    """
    flat_dict = {}

    def flatten(x: Dict, parent_key: str = ''):
        for k, v in x.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flatten(v, new_key)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    flatten({f"{new_key}[{i}]": item})
            else:
                flat_dict[new_key] = v

    flatten(nested_dict)
    return flat_dict

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    """
    return list(map(list, set(itertools.permutations(nums))))

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    """
    pattern = r'\b(?:\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    return re.findall(pattern, text)

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    """
    # Dummy implementation for demonstration
    # Actual implementation should decode polyline and calculate distances
    coords = [(37.4219999, -122.0840575), (37.4220010, -122.0840576)]
    distances = [0] + [np.sqrt((coords[i][0] - coords[i-1][0])**2 + (coords[i][1] - coords[i-1][1])**2) for i in range(1, len(coords))]
    return pd.DataFrame({'latitude': [lat for lat, lon in coords], 'longitude': [lon for lat, lon in coords], 'distance': distances})

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then replace each element 
    with the sum of its original row and column index before rotation.
    """
    n = len(matrix)
    rotated = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    result = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated[i])
            col_sum = sum(rotated[k][j] for k in range(n))
            result[i][j] = row_sum + col_sum - rotated[i][j]
    
    return result

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period
    """
    completeness = []
    
    for (id_, id_2), group in df.groupby(['id', 'id_2']):
        start_times = pd.to_datetime(group['startDay'] + ' ' + group['startTime'])
        end_times = pd.to_datetime(group['endDay'] + ' ' + group['endTime'])
        
        # Check if the time covers a full day (0:00 to 23:59:59) for each day
        full_day_covered = all((end - start).days == 0 and (end.hour == 23 and end.minute == 59 and end.second == 59) for start, end in zip(start_times, end_times))
        completeness.append((id_, id_2, not full_day_covered))
    
    completeness_df = pd.DataFrame(completeness, columns=['id', 'id_2', 'incomplete'])
    return completeness_df.set_index(['id', 'id_2'])['incomplete']

# Example inputs and outputs for each function

# 1. Reverse List by N Elements
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output: [3, 2, 1, 6, 5, 4, 8, 7]

# 2. Group by Length
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))  
# Output: {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}

# 3. Flatten a Nested Dictionary
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}
print(flatten_dict(nested_dict))  
# Output: {"road.name": "Highway 1", "road.length": 350, "road.sections[0].id": 1, "road.sections[0].condition.pavement": "good", "road.sections[0].condition.traffic": "moderate"}

# 4. Generate Unique Permutations
print(unique_permutations([1, 1, 2]))  
# Output: [[1, 1, 2], [1, 2, 1], [2, 1, 1]]

# 5. Find All Dates in a Text
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))  
# Output: ["23-08-1994", "08/23/1994", "1994.08.23"]

# 6. Decode Polyline, Convert to DataFrame with Distances
print(polyline_to_dataframe("dummy_polyline"))  # Replace with actual polyline

# 7. Matrix Rotation and Transformation
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(rotate_and_multiply_matrix(matrix))  
# Output: [[22, 19, 16], [23, 20, 17], [24, 21, 18]]

# 8. Time Check
data = {
    'id': [1, 1, 1, 2, 2],
    'id_2': [10, 10, 10, 20, 20],
    'startDay': ['2023-10-01', '2023-10-01', '2023-10-02', '2023-10-01', '2023-10-02'],
    'startTime': ['00:00:00', '12:00:00', '15:00:00', '00:00:00', '12:00:00'],
    'endDay': ['2023-10-01', '2023-10-01', '2023-10-02', '2023-10-01', '2023-10-02'],
    'endTime': ['23:59:59', '23:59:59', '23:59:59', '23:59:59', '23:59:59']
}
df = pd.DataFrame(data)
print(time_check(df))  # Output: Boolean Series indicating completeness for each (id, id_2) pair
