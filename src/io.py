import os
from typing import List

def fetch_image_paths(data_path: str) -> List[str]:
    tmp_li = []
    if not os.path.exists(data_path): return []
    for item in os.listdir(data_path):
        tmp_li.append(os.path.join(data_path, item))
    return tmp_li

def save_img(img):
    pass
