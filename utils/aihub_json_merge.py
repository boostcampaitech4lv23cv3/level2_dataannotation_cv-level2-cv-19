import os
import json
from json_handle import xywh2xyxy


def json_collector(src_path: str, indent=False):
    output = {"images": {}}
    dir_name = src_path.split(os.sep)[-1]
    for file in os.listdir(src_path):
        if file.endswith(".json"):
            with open(os.path.join(src_path, file), 'r+', encoding='utf-8') as f:
                dic = json.load(f)
            output["images"][str(dic["images"][0]["file_name"])] = {
                "img_h": dic["images"][0]["height"],
                "img_w": dic["images"][0]["width"],
                "words": {
                    items["id"]: {
                        "points": xywh2xyxy(items["bbox"]),
                        "transcription": items["text"],
                        "language": ["ko"],
                        "illegibility": False if items["text"] != "xxx" else True,
                        "orientation": "Horizontal" if dic["metadata"][0]["wordorientation"] == "가로" else "Vertical",
                        # "word_tags": dic["metadata"]
                        "word_tags": []
                    } for items in dic["annotations"]
                }
            }
    if not indent:
        with open(dir_name + ".json", "w+", encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False)
    else:
        with open(dir_name + "_indent.json", "w+", encoding='utf-8') as f:
            json.dump(output, f, indent=indent, ensure_ascii=False)


if __name__ == "__main__":
    LABEL_PATH = r'C:\Users\Joel\Desktop\야외 실제 촬영 한글 이미지\Training\[라벨]Training'
    target_paths = set()


    def path_finder(src_path: str):
        for fn in os.listdir(src_path):
            item_path = os.path.join(src_path, fn)
            if os.path.isfile(item_path) and fn.endswith(".json"):
                target_paths.add(src_path)
            if os.path.isdir(item_path):
                path_finder(item_path)


    path_finder(LABEL_PATH)

    for item in sorted(list(target_paths)):
        # print(item)
        json_collector(item, indent=False)
