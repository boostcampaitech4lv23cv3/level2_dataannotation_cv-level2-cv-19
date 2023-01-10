import os.path as osp
from os import listdir, walk
import shutil
import json

ROOT = "C:/Users/Joel/Desktop/야외 실제 촬영 한글 이미지/"
DATASET_PATH = osp.join(ROOT, "Training")

json_lists = [fn for fn in listdir(ROOT) if fn.endswith(".json")]
for j in json_lists:
    with open(osp.join(ROOT, j), "r+", encoding='utf-8') as f:
        json = json.load(f)
    print(j, len(json['images']), json['images'].keys())
    break