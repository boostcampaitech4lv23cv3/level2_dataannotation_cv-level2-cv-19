import os
import shutil
import json
import sys
import numpy as np
from unidecode import unidecode

from PIL import Image
from datetime import datetime, timedelta


LANGUAGE_MAP = {
    'Korean': 'ko',
    'Latin': 'en',
    'Symbols': None,
    'None': None
}


def get_language_token(x):
    return LANGUAGE_MAP.get(x, 'others')


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
                    language = get_language_token(items[8])
                    if language == "en":
                        transcription = unidecode(','.join(items[9:]))
                    else:
                        transcription = ','.join(items[9:])
                    points = np.array(items[:8], dtype=np.float32).reshape(4, 2).tolist()

                    word_info[idx] = {
                        "transcription": transcription if transcription != "###" else None,
                        "points": points,
                        "orientation": "Horizontal",
                        "language": [language],
                        "illegibility": False if transcription != "###" else True,
                        # "word_tags": dic["metadata"]
                        "word_tags": []
                    }
                    output["images"][img]["words"].update(word_info)

                output["images"][img]["width"], output["images"][img]["height"] = Image.open(os.path.join(src_img_path, img)).size
                # output["images"][img]["annotation_log"] = {"worker": "Je-won", "timestamp": datetime.strftime(datetime.now() + timedelta(hours=9), "%Y-%m-%d"), "source": "ICDAR17"}
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
        dst = os.path.join(src_path, f"{name}_{str(idx).zfill(5)}.{ext}")
        os.rename(src, dst)


if __name__ == "__main__":
    ROOT = "./"
    train_img_path = os.path.join(ROOT, "images")
    train_anno_path = os.path.join(ROOT, "labels")
    val_img_path = os.path.join(ROOT, "ch8_validation_images")
    val_anno_path = os.path.join(ROOT, "ch8_validation_localization_transcription_gt_v2")

    # change_filenames(train_img_path)
    # change_filenames(train_anno_path)
    convert(src_img_path=train_img_path, src_label_path=train_anno_path, dst_path=ROOT, is_val=False, indent=None)
    # change_filenames(val_img_path, offset=10000)
    # change_filenames(val_anno_path, offset=10000)
    convert(src_img_path=val_img_path, src_label_path=val_anno_path, dst_path=ROOT, is_val=True, indent=None)
