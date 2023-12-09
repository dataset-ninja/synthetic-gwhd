# https://www.kaggle.com/datasets/bendang/synthetic-wheat-images

import ast
import csv
import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "synthetic wheat"
    images_path = "/home/grokhi/rawdata/synthetic-gwhd/images"
    corrected_path = "/home/grokhi/rawdata/synthetic-gwhd/corrected_train.csv"
    pix2pix_1_path = "/home/grokhi/rawdata/synthetic-gwhd/pix2pix_1_synthetic.csv"
    pix2pix_2_path = "/home/grokhi/rawdata/synthetic-gwhd/pix2pix_2_synthetic.csv"
    style_path = "/home/grokhi/rawdata/synthetic-gwhd/style_transfer_images.csv"

    batch_size = 30
    ds_name = "ds"

    def create_ann(image_path):
        labels = []
        tags = []

        # image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = 1024  # image_np.shape[0]
        img_wight = 1024  # image_np.shape[1]

        # file_value = im_name_to_file.get(get_file_name(image_path))
        # if file_value is not None:
        #     file_tag = sly.Tag(file_meta, value=file_value)
        #     tags.append(file_tag)

        source_value = im_name_to_source.get(get_file_name(image_path))
        if source_value is not None:
            source = sly.Tag(source_meta, value=source_value)
            tags.append(source)

        bboxes_data = im_name_to_data.get(get_file_name(image_path))

        if bboxes_data is not None:
            for curr_data in bboxes_data:
                left = curr_data[0]
                right = curr_data[0] + curr_data[2]
                top = curr_data[1]
                bottom = curr_data[1] + curr_data[3]
                if left > right or top > bottom:
                    continue
                rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                label = sly.Label(rectangle, obj_class)
                labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    obj_class = sly.ObjClass("wheat", sly.Rectangle)
    source_meta = sly.TagMeta("source", sly.TagValueType.ANY_STRING)
    # file_meta = sly.TagMeta("ann file name", sly.TagValueType.ANY_STRING)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[obj_class], tag_metas=[source_meta])
    api.project.update_meta(project.id, meta.to_json())

    im_name_to_data = defaultdict(list)
    im_name_to_file = {}
    im_name_to_source = {}
    for ann_path in [corrected_path, pix2pix_1_path, pix2pix_2_path, style_path]:
        with open(ann_path, "r") as file:
            csvreader = csv.reader(file)
            for idx, row in enumerate(csvreader):
                if idx == 0:
                    continue
                im_name_to_data[row[0]].append(ast.literal_eval(row[3]))
                if row[0] not in im_name_to_file.keys():
                    im_name_to_file[row[0]] = get_file_name(ann_path)
                if row[0] not in im_name_to_source.keys():
                    im_name_to_source[row[0]] = row[4]

    regrouped_dict = {}
    for key, value in im_name_to_file.items():
        if value not in regrouped_dict:
            regrouped_dict[value] = [key]
        else:
            regrouped_dict[value].append(key)

    for ds_name, images_names in regrouped_dict.items():
        images_names = [
            f"{name}.jpg" for name in images_names if os.path.exists(f"{images_path}/{name}.jpg")
        ]

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [
                os.path.join(images_path, image_name) for image_name in images_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))
    return project
