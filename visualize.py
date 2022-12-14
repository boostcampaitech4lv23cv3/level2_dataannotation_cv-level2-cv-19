import numpy as np
from dataset import get_rotate_mat
import matplotlib.pyplot as plt
import lanms
import cv2

def draw_bbox(image, bbox, color=(0, 0, 255), thickness=1, thickness_sub=None, double_lined=False,
              write_point_numbers=False):
    """이미지에 하나의 bounding box를 그려넣는 함수
    """
    thickness_sub = thickness_sub or thickness * 3
    basis = max(image.shape[:2])
    fontsize = basis / 1500
    x_offset, y_offset = int(fontsize * 12), int(fontsize * 10)
    color_sub = (255 - color[0], 255 - color[1], 255 - color[2])

    points = [(int(np.rint(p[0])), int(np.rint(p[1]))) for p in bbox]

    for idx in range(len(points)):
        if double_lined:
            cv2.line(image, points[idx], points[(idx + 1) % len(points)], color_sub,
                     thickness=thickness_sub)

        cv2.line(image, points[idx], points[(idx + 1) % len(points)], color, thickness=thickness)

    if write_point_numbers:
        for idx in range(len(points)):
            loc = (points[idx][0] - x_offset, points[idx][1] - y_offset)
            if double_lined:
                cv2.putText(image, str(idx), loc, cv2.FONT_HERSHEY_SIMPLEX, fontsize, color_sub,
                            thickness_sub, cv2.LINE_AA)
            cv2.putText(image, str(idx), loc, cv2.FONT_HERSHEY_SIMPLEX, fontsize, color, thickness,
                        cv2.LINE_AA)

def draw_bboxes(image, bboxes, color=(0, 0, 255), thickness=1, thickness_sub=None,
                double_lined=False, write_point_numbers=False):
    """이미지에 다수의 bounding box들을 그려넣는 함수
    """
    for bbox in bboxes:
        draw_bbox(image, bbox, color=color, thickness=thickness, thickness_sub=thickness_sub,
                  double_lined=double_lined, write_point_numbers=write_point_numbers)

def drawimgbbox(image, score_map, geo_map, orig_size):
    visualize_single_sample = True
    SAMPLE_POINT_IDX = 200

    NMS_THRES = 0.2
    SCORE_THRES = 0.9

    MAP_SCALE = 0.25
    INV_MAP_SCALE = int(1 / MAP_SCALE)

    FIG_SIZE = (8, 8)

    INPUT_SIZE = 1024

    xy_text = np.argwhere(score_map > SCORE_THRES)[:, ::-1].copy()  # (n x 2)

    if xy_text.size == 0:
        bboxes = np.zeros((0, 4, 2), dtype=np.float32)
    else:
        xy_text = xy_text[np.argsort(xy_text[:, 1])]  # Row-wise로 정렬
        valid_pos = xy_text * INV_MAP_SCALE
        valid_geo = geo_map[xy_text[:, 1], xy_text[:, 0], :]  # (n x 5)
        
        indices, bboxes = [], []
        for idx, ((x, y), g) in enumerate(zip(valid_pos, valid_geo)):
            y_min, y_max = y - g[0], y + g[1]
            x_min, x_max = x - g[2], x + g[3]
            rotate_mat = get_rotate_mat(-g[4])
            
            bbox = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
            anchor = np.array([x, y], dtype=np.float32).reshape(2, 1)
            rotated_bbox = (np.dot(rotate_mat, bbox.T - anchor) + anchor).T
            
            if visualize_single_sample and idx == SAMPLE_POINT_IDX:
                vis = score_map.copy() * 255
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB).astype(np.uint8)
                draw_bboxes(vis, bbox.reshape(-1, 4, 2) * MAP_SCALE, thickness=2)
                draw_bboxes(vis, rotated_bbox.reshape(-1, 4, 2) * MAP_SCALE, color=(0, 255, 0), thickness=2)
                ref_point = np.array([x, y]) * MAP_SCALE
                cv2.circle(vis, (int(ref_point[0]), int(ref_point[1])), 2, (255, 0 ,0), 2)
                plt.figure(figsize=FIG_SIZE)
                plt.imshow(vis)
            
            # 이미지 범위에서 벗어나는 bbox는 탈락
            if bbox[:, 0].min() < 0 or bbox[:, 0].max() >= score_map.shape[1] * INV_MAP_SCALE:
                continue
            elif bbox[:, 1].min() < 0 or bbox[:, 1].max() >= score_map.shape[0] * INV_MAP_SCALE:
                continue

            indices.append(idx)
            bboxes.append(rotated_bbox.flatten())
        bboxes = np.array(bboxes)
        
        raw_bboxes = bboxes.reshape(-1, 4, 2)
        
        # 좌표 정보에 Score map에서 가져온 Score를 추가
        scored_bboxes = np.zeros((bboxes.shape[0], 9), dtype=np.float32)
        scored_bboxes[:, :8] = bboxes
        scored_bboxes[:, 8] = score_map[xy_text[indices, 1], xy_text[indices, 0]]
        
        # LA-NMS 적용
        nms_bboxes = lanms.merge_quadrangle_n9(scored_bboxes.astype('float32'), NMS_THRES)
        nms_bboxes = nms_bboxes[:, :8].reshape(-1, 4, 2)
        
        # 원본 이미지 크기에 맞게 bbox 크기 보정
        raw_bboxes *= max(orig_size) / INPUT_SIZE
        nms_bboxes *= max(orig_size) / INPUT_SIZE

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(FIG_SIZE[0] * 2, FIG_SIZE[1]))
    plt.suptitle('Before & After NMS Process', fontsize=18, y=0.95)

    vis = image.copy()
    draw_bboxes(vis, raw_bboxes[::], thickness=2)
    axs[0].imshow(vis)

    vis = image.copy()
    draw_bboxes(vis, nms_bboxes, thickness=2)
    axs[1].imshow(vis)