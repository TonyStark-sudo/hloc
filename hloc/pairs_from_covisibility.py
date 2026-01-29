import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import logger
from .utils.read_write_model import read_model


def main(model, output, num_matched):
    logger.info("Reading the COLMAP model...")
    cameras, images, points3D = read_model(model)

    logger.info("Extracting image pairs from covisibility info...")
    pairs = []
    for image_id, image in tqdm(images.items()):
        # np.array ！= -1 是NumPy的向量化比较操作，返回一个布尔数组，其中 ！= -1 的元素为 True
        matched = image.point3D_ids != -1
        # 这是 NumPy 的一个核心功能，叫做布尔索引，会返回一个新的数组，这个新数组只包含原始数组中与 matched 数组中 True 值相对应的那些元素。
        points3D_covis = image.point3D_ids[matched]

        # 找每个3d点的共视图像
        covis = defaultdict(int)
        for point_id in points3D_covis:
            for image_covis_id in points3D[point_id].image_ids:
                if image_covis_id != image_id:
                    covis[image_covis_id] += 1

        if len(covis) == 0:
            logger.info(f"Image {image_id} does not have any covisibility.")
            continue

        covis_ids = np.array(list(covis.keys()))
        covis_num = np.array([covis[i] for i in covis_ids])

        if len(covis_ids) <= num_matched:
            top_covis_ids = covis_ids[np.argsort(-covis_num)]
        else:
            # get covisible image ids with top k number of common matches
            ind_top = np.argpartition(covis_num, -num_matched)
            ind_top = ind_top[-num_matched:]  # unsorted top k
            ind_top = ind_top[np.argsort(-covis_num[ind_top])]
            top_covis_ids = [covis_ids[i] for i in ind_top]
            assert covis_num[ind_top[0]] == np.max(covis_num)

        for i in top_covis_ids:
            pair = (image.name, images[i].name)
            pairs.append(pair)

    logger.info(f"Found {len(pairs)} pairs.")
    print(pairs[:10])
    # Special-case output formatting for some pipelines that expect
    # paths prefixed with `db/` (e.g. `pairs-ourdb-covis20.txt`).
    out_name = Path(output).name
    with open(output, "w") as f:
        if out_name == "pairs-ourdb-covis20.txt":
            # prepend `db/` to both entries per line
            f.write("\n".join(f"db/{i} db/{j}" for i, j in pairs))
        else:
            f.write("\n".join(" ".join([i, j]) for i, j in pairs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num_matched", required=True, type=int)
    args = parser.parse_args()
    main(**args.__dict__)
