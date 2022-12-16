import os
import shutil
import json
import sys

import numpy as np
from PIL import Image
from datetime import datetime, timedelta

ROOT = "./"

train_img_path = os.path.join(ROOT, "ch4_training_images")
train_anno_path = os.path.join(ROOT, "ch4_training_localization_transcription_gt")
test_img_path = os.path.join(ROOT, "ch4_test_images")
test_anno_path = os.path.join(ROOT, "Challenge4_Test_Task1_GT")


def create_path(target_path: str) -> None:
    if not os.path.exists(target_path):
        os.makedirs(target_path)


def convert(src_img_path, src_label_path, dst_path, is_val=False, indent=None) -> bool:
    try:
        output = {"images": {}}
        for img, label in zip(os.listdir(src_img_path), os.listdir(src_label_path)):
            if label.endswith(".txt"):
                with open(os.path.join(src_label_path, label), 'r+', encoding='utf-8-sig') as f:
                    lines = f.readlines()
                output["images"][img] = {}
                output["images"][img]["words"] = {}
                for idx, line in enumerate(lines, 1):
                    word_info = {}
                    items = line.strip().split(",")
                    idx = str(idx).zfill(4)
                    transcription = ','.join(items[8:])
                    points = np.array(items[:8], dtype=np.float32).reshape(4, 2).tolist()

                    word_info[idx] = {
                        "transcription": transcription if transcription != "###" else None,
                        "points": points,
                        "orientation": "Horizontal",
                        "language": ["en"],
                        "illegibility": False if transcription != "###" else True,
                        # "word_tags": dic["metadata"]
                        "word_tags": []
                    }
                    output["images"][img]["words"].update(word_info)

                output["images"][img]["width"], output["images"][img]["height"] = Image.open(
                    os.path.join(src_img_path, img)).size
                output["images"][img]["annotation_log"] = {"worker": "Je-won",
                                                           "timestamp": datetime.strftime(
                                                               datetime.now() + timedelta(hours=9), "%Y-%m-%d"),
                                                           "source": "ICDAR15"}
        dst_path = os.path.join(dst_path, "ufo")
        create_path(dst_path)
        with open(os.path.join(dst_path, "validation.json" if is_val else "train.json"), "w+", encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=indent)
        return True
    except Exception as e:
        print("Error", e.with_traceback(sys.exc_info()[2]))
        return False


def change_filenames(src_path, offset=0):
    print(src_path, "Changing FN")
    for file in os.listdir(src_path):
        src = os.path.join(src_path, file)
        fn, ext = file.split(".")
        item = fn.split("_")
        i = int(item[-1])
        name = "_".join(item[:-1])
        idx = i + offset
        dst = os.path.join(src_path, f"{name}_{str(idx).zfill(4)}.{ext}")
        os.rename(src, dst)


if __name__ == "__main__":
    dst_path = os.path.join(ROOT, "images")
    create_path(dst_path)

    change = False
    if change:
        change_filenames(train_img_path)
        change_filenames(train_anno_path)
        cnt_offset = len(os.listdir(train_img_path))
        change_filenames(test_img_path, offset=cnt_offset)
        change_filenames(test_anno_path, offset=cnt_offset)
        # change_filenames(test_img_path, offset=-3000)
        # change_filenames(test_anno_path, offset=-3000)

    if convert(train_img_path, train_anno_path, dst_path=ROOT, is_val=False, indent=None):
        shutil.copytree(train_img_path, dst_path, dirs_exist_ok=True)

    if convert(test_img_path, test_anno_path, dst_path=ROOT, is_val=True, indent=None):
        shutil.copytree(test_img_path, dst_path, dirs_exist_ok=True)
