import os
import numpy as np
from PIL import Image
import multiprocessing
import sys
from math import sqrt
import json


def drd_weights(n: int = 2) -> np.ndarray:
    m = 2 * n + 1
    i_c, j_c = (m - 1) // 2, (m - 1) // 2
    weights = np.array([
            [1 / sqrt((i - i_c) ** 2 + (j - j_c) ** 2) if abs(j - j_c) + abs(i - i_c) != 0 else 0 for j in range(m)]
            for i in range(m)
    ])
    assert weights[i_c, j_c] == 0
    return weights / np.sum(weights)


def drd(im_gt: np.ndarray, im_pred: np.ndarray, n: int = 2, block_size: int = 8):
    weights = drd_weights(n)

    error_map = np.abs(im_gt.astype('int32') - im_pred.astype('int32'))  # //255
    height, width = im_pred.shape
    nubn = 0.

    for x1 in range(0, height, block_size):
        for y1 in range(0, width, block_size):
            if x1 + block_size > height or y1 + block_size > width:
                break
            x2 = x1 + block_size
            y2 = y1 + block_size

            block_dim = (y2 - y1) * (x2 - x1)
            block_sum = np.sum(im_gt[x1: x2, y1: y2])

            if block_sum != 0 and block_sum < block_dim:
                nubn += 1

    drd_sum = 0
    for i in range(error_map.shape[0]):
        for j in range(error_map.shape[1]):
            if error_map[i][j] != 0:
                x_max = min(error_map.shape[0], i + n + 1)
                y_max = min(error_map.shape[1], j + n + 1)
                x_min = max(0, i - n)
                y_min = max(0, j - n)

                wx = i + n + 1 - x_max
                wy = j + n + 1 - y_max
                hx = abs(i - n - x_min)
                hy = abs(j - n - y_min)
                map_gt = im_gt[x_min: x_max, y_min: y_max].copy()
                if im_gt[i][j] == 0:
                    map_gt = -map_gt.astype('int32') + 1

                drd_sum += np.sum(map_gt * weights[hx: 2 * n + 1 - wx, hy: 2 * n + 1 - wy])

    if nubn == 0:
        nubn = 1
        print("Number of non-empty blocks is zero. NUNB = 1.")

    return drd_sum / nubn


def read_img(image_file, mask_file):
    img = np.array(Image.open(image_file))
    mask = np.array(Image.open(mask_file))

    return np.array(img), np.array(mask)


def metrics(img):
    line, mask_path = img
    pred, gt = read_img(line, mask_path)

    return drd(gt, pred)


def get_json(results, out_path, author=' ', version='1.1'):
    output_json = dict()
    output_json['author'] = author
    output_json['version'] = version
    all_num = 0
    DRD_all = 0
    output_json['stands'] = []
    for DRD, num_img, name in results:
        all_num += num_img
        DRD_all += DRD * num_img
        stend_info = dict()
        stend_info['name'] = name
        stend_info['num_of_images'] = num_img
        stend_info['metrics'] = dict()
        stend_info['metrics']['DRD'] = DRD
        output_json['stands'].append(stend_info)
    #print(np.mean(DRDk))
    output_json['all'] = dict()
    output_json['all']['num_of_images'] = all_num
    output_json['all']['metrics'] = dict()
    output_json['all']['metrics']['DRD'] = DRD_all / all_num
    with open(os.path.join(out_path, 'stat.json'), "w") as write_file:
        json.dump(output_json, write_file, indent=2)


def calculate(path_to_lst, out_dir):
    #print(path_to_lst)
    #PurePath(path_to_lst).parent
    results = []

    with open(path_to_lst, "r") as f:
        for line in f:
            dir_pt = os.path.dirname(path_to_lst)
            dataset = os.path.basename(os.path.dirname(line))
            print(dataset)
            images_lst_path = str(dir_pt) + str(line)
            out_dir_cur = os.path.dirname(out_dir + line)
            path_gt = str(os.path.dirname(images_lst_path)) + '/_ground_truth/gt'

            #print(path_gt)

            if images_lst_path[-9:] == 'local.lst':
                path_gt += '_local.lst'
            else:
                path_gt += '.lst'
            with open(path_gt, 'r') as fp:
                count = len(fp.readlines())

            imgs = []
            ind = 0
            l = len(path_gt) - 1
            while path_gt[l] != '/':
                l -= 1
            path_to_img = path_gt[:l + 1]
            with open(path_gt, "r") as file1:
                for line in file1:
                    gt = path_to_img + str(line)[:-1]
                    predict = os.path.join(out_dir_cur, os.path.splitext(line)[0]) + '.png'
                    imgs.append((predict, gt))

            pool = multiprocessing.Pool()
            DRDk = pool.map(metrics, imgs)
            pool.close()
            pool.join()
            results.append((np.mean(DRDk), len(imgs), dataset))

    get_json(results, out_dir)


if __name__ == '__main__':
    cfg_path = sys.argv[1]
    out_dir = sys.argv[2]

    with open(cfg_path, "r") as read_file:
        cfg = json.load(read_file)
        path_to_lst = cfg['path_to_lst']

        calculate(path_to_lst, out_dir)
