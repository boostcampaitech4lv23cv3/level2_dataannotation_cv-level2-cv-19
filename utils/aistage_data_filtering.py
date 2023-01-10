import os.path as osp
import json
from copy import deepcopy

root_dir='../input/data/aistages'
with open(osp.join(root_dir, 'ufo/annotation.json'), 'r+') as f:
    anno = json.load(f)

image_fnames = sorted(anno['images'].keys())
image_dir = osp.join(root_dir, 'images')

# new_json = deepcopy(anno)
for main_keys in anno.copy().keys():
    if main_keys == "images":
        for file_name in anno[main_keys].copy().keys():
            for data_type in anno[main_keys][file_name].copy().keys():
                if data_type == "words":
                    for word_idx in anno[main_keys][file_name][data_type].copy().keys():
                        if "transcription" not in anno[main_keys][file_name][data_type][word_idx].keys() or anno[main_keys][file_name][data_type][word_idx]["transcription"] == "" or anno[main_keys][file_name][data_type][word_idx]["transcription"] is None:
                            del anno[main_keys][file_name][data_type][word_idx]
                        elif "points" not in anno[main_keys][file_name][data_type][word_idx].keys() or len(anno[main_keys][file_name][data_type][word_idx]["points"]) != 4:
                            del anno[main_keys][file_name][data_type][word_idx]
            if anno[main_keys][file_name]["words"] == {} or "words" not in anno[main_keys][file_name].keys():
                del anno[main_keys][file_name]

with open(osp.join(root_dir, 'ufo/annotation_0.json'), 'w+') as f:
    json.dump(anno, f, indent=4, ensure_ascii=False)