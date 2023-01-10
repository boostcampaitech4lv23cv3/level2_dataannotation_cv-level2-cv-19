import os.path as osp
from os import listdir, walk
import shutil
import json

ROOT = "C:/Users/Joel/Desktop/야외 실제 촬영 한글 이미지/"
DATASET_PATH = osp.join(ROOT, "Training")
divide_into = 12

output_json = {"images": {}}

json_lists = [fn for fn in listdir(ROOT) if fn.endswith(".json")]
for j in json_lists:
    try:
        with open(osp.join(ROOT, j), "r+", encoding='utf-8') as f:
            jsons = json.load(f)
        output_json["images"].update({key: value for key, value in list(jsons['images'].items())[::divide_into]})
    except:
        print(j)

with open(osp.join(ROOT, "train.json"), 'w+', encoding='utf-8') as f:
    json.dump(output_json, f, ensure_ascii=False)
