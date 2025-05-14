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

# %% jupyter={"is_executing": true}
import os
from typing import Optional

import ms3
import pandas as pd

from AugmentedNet.common import ANNOTATIONSCOREDUPLES, DATASPLITS
from AugmentedNet.dataset_tsv_generator import store_labeled_pitch_array_and_label_tsv
from AugmentedNet.joint_parser import parseAnnotationAndScoreEvents

os.chdir("..")

# %%
# %load_ext autoreload
# %autoreload 2
from AugmentedNet import utils


# %%
def assert_folder(path, show=True):
    abs_path = ms3.resolve_dir(path)
    assert os.path.isdir(abs_path), f"Note a directory: {abs_path}"
    if show:
        print(abs_path)
    return abs_path


AUGNET_DATASET = assert_folder("events")
DLC_DATASET = assert_folder("~/distant_listening_corpus/processing")

# %%
merged_summary = pd.read_csv("merged_summary.tsv", sep="\t")
excluded = merged_summary.corpus_dlc.notna() & (merged_summary.split_v100 == "test")
print(f"{excluded.sum()} pieces excluded from DLC because they are in AugNet test set.")
excluded_ids = list(merged_summary.loc[excluded, ["corpus_dlc", "piece"]].itertuples(index=False, name=None))
excluded_ids.extend([
    ("chopin_mazurkas", "BI61-5op07-5"),
    ("chopin_mazurkas", "BI77-3op17-3")
])
merged_summary = merged_summary.loc[~excluded]
merged_summary

# %%
excluded_nicknames = [f"{c}_{p}" for c, p in excluded_ids]
excluded_nicknames


# %% [markdown]
# ## Overview

# %%
def get_piece_index(df):
    return df.index.get_level_values(0).unique()

get_piece_index(augnet)


# %%
get_piece_index(dlc)

# %% [markdown]
# ## DLC

# %%
metadata_path = os.path.join(DLC_DATASET, "distant_listening_corpus.metadata.tsv")
dlc_metadata = ms3.load_tsv(metadata_path)


def make_nickname_index(dlc_metadata):
    return pd.Index(dlc_metadata.corpus + "_" + dlc_metadata.piece, name="nickname")


dlc_metadata.index = make_nickname_index(dlc_metadata)
dlc_metadata = dlc_metadata.loc[dlc_metadata.index.difference(excluded_nicknames)]
dlc_metadata

# %%
dlc_annotations = ms3.load_tsv("/home/laser/Documents/Linz/DLC_version_comparison/distant_listening_corpus_v3.1/distant_listening_corpus.expanded.tsv")
dlc_annotations.index = make_nickname_index(dlc_annotations)
dlc_annotations = dlc_annotations.loc[dlc_metadata.index].copy()
dlc_annotations


# %%
def count_labels(dlc_annotations, piece_wise=False):
    labels = pd.concat([
        dlc_annotations.chord.notna(),
        dlc_annotations.cadence.notna(),
        dlc_annotations.phraseend.isin([r"\\", "}", "}{"]).rename("phrase")
    ], axis=1)
    if piece_wise:
        return labels.groupby(level=0).sum()
    return labels.sum()

count_labels(dlc_annotations)

# %%
dlc_metadata[dlc_metadata.piece.duplicated(keep=False)]

# %%
dlc_unfolded_annotations = ms3.load_tsv(os.path.join(DLC_DATASET, "..", "unfolded_harmonies", "unfolded_harmonies.expanded.tsv"))
dlc_unfolded_annotations

# %%
dlc.groupby(level=0)[["valid_chord_label", "valid_cadence_label", "valid_phrase_label", "valid_pedal_point_label"]].agg(["size", "sum"])

# %%
dlc_summary = merged_summary[merged_summary.corpus_dlc.notna()]
dlc_summary[["has_chords", "has_cadence", "has_phrase", "has_pedal"]].sum()


# %%
# with_duplicates = pd.concat(dict(
#     DLC = merged_summary[merged_summary.corpus_dlc.notna()],
#     AugmentedNet = merged_summary[merged_summary.split_v100.notna()],
# )).reset_index(level=0, names="dataset")
# # the boolean columns are not correct for AugNet and Augnet is missing the excluded pieces here;
# # it's simply has_chords = 353 (# pieces), and 0 for the rest
# with_duplicates.groupby("dataset")[["has_chords", "has_cadence", "has_phrase", "has_pedal"]].sum()

# %% [markdown]
# ## The Concatenation

# %%
def concat_augnet_arrays(dataset_path, specs_path):
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
            filepath = os.path.join(split_dir, file)
            pitch_arrays[piece] = utils.load_labeled_pitch_array(
                specs_csv=specs_path,
                pitch_array_tsv=filepath,
            )
    return pd.concat(pitch_arrays)

specs_path = "augnet_pitch_array_specs.csv"
augnet = concat_augnet_arrays(AUGNET_DATASET, specs_path)


# %%
def concat_dlc_arrays(dataset_path, specs_path):
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
            filepath = os.path.join(corpus_dir, file)
            pitch_arrays[nickname] = utils.load_labeled_pitch_array(
                specs_csv=specs_path,
                pitch_array_tsv=filepath,
            )
    return pd.concat(pitch_arrays)

specs_path = os.path.join(DLC_DATASET, "dlc_pitch_array_specs.csv")
pitch_arrays_dir = os.path.join(DLC_DATASET, "pitch_arrays")
dlc = concat_dlc_arrays(pitch_arrays_dir, specs_path)

# %% [markdown]
# ## Checking columns and values

# %%
grouped = augnet.note_degree.isna().groupby(augnet.index.get_level_values(0)).sum()
grouped[grouped > 0]

# %%
numeral_counts = augnet.a_simpleNumeral.value_counts(dropna=False)
numeral_counts[["Cad"]]

# %%
numeral_counts.sum()

# %%
tonic_inversion_mask = (augnet.a_degree1 == "1") & (augnet.a_inversion == "2")
tonic_inversion_mask.sum()

# %%
ti_subsequent_mask = tonic_inversion_mask.where(tonic_inversion_mask, tonic_inversion_mask.shift())
augnet[ti_subsequent_mask]


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


store_individual_pitch_array("tavern-beethoven-woo-75-b", "events", "assembled")

# %%
dlc.note_degree.value_counts()

# %%
cadential_V_mask.sum() / dlc.chord.notna().sum()

# %%
dlc.to_csv("dlc_pitch_arrays.tsv", sep="\t", index=False)

# %%
dlc.columns

# %%
dlc.valid_pedal_point_label.value_counts(dropna=False)

# %%
dlc.a_bass.value_counts()

# %%
from pprint import pprint

def make_vocab_set(augnet_column, fillna):
    return set(augnet_column.unique().fillna(fillna))

def get_vocabulary_from_series(dlc_column, augnet_column, fillna="None"):
    dlc_vocab = make_vocab_set(dlc_column, fillna)
    augnet_vocab = make_vocab_set(augnet_column, fillna)
    joint = dlc_vocab.union(augnet_vocab)
    print(len(joint))
    return joint

def get_vocabulary(dlc, augnet, dlc_col, augnet_col=None, fillna="None"):
    if not augnet_col:
        augnet_col = dlc_col
    dlc_column = dlc[dlc_col]
    augnet_column = augnet[augnet_col]
    joint = get_vocabulary_from_series(dlc_column, augnet_column, fillna)
    return joint



get_vocabulary(dlc, augnet, "a_root")

# %%
root = get_vocabulary_from_series(dlc.a_root, augnet.a_root.str.replace("-", "b"))

# %%
bass = get_vocabulary_from_series(dlc.a_bass, augnet.a_bass.str.replace("-", "b"))

# %%
joint_tone_functions = root.union(bass)
len(joint_tone_functions)
joint_tone_functions

# %%
keys = get_vocabulary_from_series(dlc.a_localKey, augnet.a_localKey.str.replace("-", "b"))

# %%
tonicized = get_vocabulary_from_series(dlc.a_tonicizedKey, augnet.a_tonicizedKey.str.replace("-", "b"))

# %%
globalkey = get_vocabulary_from_series(dlc.a_gl, augnet.a_tonicizedKey.str.replace("-", "b"))

# %%
tonicized

# %%
get_vocabulary_from_series(dlc.a_degree1, augnet.a_degree1)

# %%

sorted(get_vocabulary(dlc, augnet, "downbeat"), key=int)

# %%

augnet.ts_beats.dtype
