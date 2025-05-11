# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
from typing import Optional

import ms3
import pandas as pd

from AugmentedNet.common import ANNOTATIONSCOREDUPLES, DATASPLITS
from AugmentedNet.dataset_tsv_generator import store_labeled_pitch_array_and_label_tsv
from AugmentedNet.joint_parser import parseAnnotationAndScoreEvents

os.chdir("../../corpora/AugmentedNet")

# %%
# %load_ext autoreload
# %autoreload 2
# from AugmentedNet import utils


# %%
def assert_folder(path, show=True):
    abs_path = ms3.resolve_dir(path)
    assert os.path.isdir(abs_path), f"Note a directory: {abs_path}"
    if show:
        print(abs_path)
    return abs_path


AUGNET_DATASET = assert_folder("events")
DLC_DATASET = assert_folder("~/distant_listening_corpus/processing/pitch_arrays")


# %%
def concat_augnet_arrays(dataset_path):
    pitch_arrays = {}
    for split in os.listdir(dataset_path):
        split_dir = os.path.join(dataset_path, split)
        if os.path.isfile(split_dir):
            continue
        for file in os.listdir(split_dir):
            suffix = "_joint.tsv"
            if not file.endswith(suffix):
                continue
            piece = file[: -len(suffix)]
            print(".", end="")
            pitch_arrays[piece] = pd.read_csv(
                os.path.join(split_dir, file), sep="\t", dtype="string"
            )
    return pd.concat(pitch_arrays)


augnet = concat_augnet_arrays(AUGNET_DATASET)

# %%
augnet[augnet.valid_chord_label.isna()].index.get_level_values(0).unique()


# %%
def get_individual_pitch_array(nickname):
    annotation_path, score_path = ANNOTATIONSCOREDUPLES[nickname]
    _, _, jointdf, _ = parseAnnotationAndScoreEvents(annotation_path, score_path)
    return jointdf


def get_split_by_nickname(nickname):
    for split, names in DATASPLITS.items():
        if nickname in names:
            return split
    raise KeyError(f"Could not find {nickname}")


def store_individual_pitch_array(
    nickname,
    datasetDir,
    assembled_dir: Optional[str] = None,
    include_metadata: bool = True,
):
    annotation, score = ANNOTATIONSCOREDUPLES[nickname]
    stats = store_labeled_pitch_array_and_label_tsv(
        nickname,
        score,
        annotation,
        datasetDir,
        get_split_by_nickname(nickname),
        assembled_dir,
        include_metadata,
    )
    return stats


store_individual_pitch_array("abc-op74-4", "events", "assembled")


# %%
def concat_dlc_arrays(dataset_path):
    pitch_arrays = {}
    for c_name in os.listdir(dataset_path):
        corpus_dir = os.path.join(dataset_path, c_name)
        if os.path.isfile(corpus_dir):
            continue
        for file in os.listdir(corpus_dir):
            p_name, fext = os.path.splitext(file)
            if file.startswith(".") or fext != ".tsv":
                continue
            nickname = f"{c_name}_{p_name}"
            print(".", end="")
            pitch_arrays[nickname] = pd.read_csv(
                os.path.join(corpus_dir, file), sep="\t", dtype="string"
            )
    return pd.concat(pitch_arrays)


dlc = concat_dlc_arrays(DLC_DATASET)

# %%
dlc.to_csv("dlc_pitch_arrays.tsv", sep="\t", index=False)

# %%
dlc[dlc.valid_chord_label.isna()]

# %%
dlc.valid_pedal_point_label.value_counts(dropna=False)

# %%
dlc[dlc.valid_pedal_point_label.isna()].index.get_level_values(0).unique().tolist()
