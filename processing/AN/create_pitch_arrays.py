# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: dilemmadata
#     language: python
#     name: dilemmadata
# ---

# %%
import os.path as osp
import shutil
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


REPO_PATH = "../../corpora/AugmentedNet"
sys.path.append(REPO_PATH)
from AugmentedNet.common import DATASETSUMMARYFILE, DATASPLITS, ANNOTATIONSCOREDUPLES
from AugmentedNet.joint_parser import parseAnnotationAndScoreEvents


# %%
def store_labeled_pitch_array_and_label_tsv(
    nickname: str,
    score_path: str,
    annotation_path: str,
    datasetDir: str,
    split: str,
    assembled_dir: Optional[str] = None,
    include_metadata: bool = True,
):
    extended_adf, sdf, jointdf, metadata = parseAnnotationAndScoreEvents(
        annotation_path, score_path
    )
    for df, suffix in [(jointdf, "joint")]:  # , (sdf, "slices")]:
        outpath = osp.join(datasetDir, split, f"{nickname}_{suffix}.tsv")
        df.to_csv(outpath, sep="\t", index=False)
    if assembled_dir:
        outpath = osp.join(assembled_dir, "labels", f"{nickname}.tsv")
        extended_adf.to_csv(outpath, sep="\t", index=False)
    # copy and rename original score
    _, score_ext = osp.splitext(score_path)
    new_score_path = osp.join(datasetDir, split, f"{nickname}{score_ext}")
    shutil.copy(score_path, new_score_path)
    collection = nickname.split("-")[0]
    stats = dict(
        file=nickname,
        annotation=annotation_path,
        score=score_path,
        collection=collection,
        split=split,
    )
    if include_metadata:
        stats.update(metadata)
    return stats


def generateEventsDataset(
        augnet_repo="AugmentedNet", tsvDir="events", assembled_dir=None, include_metadata=True, reset=True
        ):
    datasetDir = tsvDir
    dataset_summary_path = osp.join(datasetDir, DATASETSUMMARYFILE)
    if reset:
        statsrecords = []
    else:
        statsrecords = pd.read_csv(dataset_summary_path, sep="\t").to_dict(orient="records")
    Path(datasetDir).mkdir(exist_ok=True)
    for split, files in DATASPLITS.items():
        Path(osp.join(datasetDir, split)).mkdir(exist_ok=True)
        for nickname in files:
            if nickname in {rec["file"] for rec in statsrecords}:
                print(f"{nickname} SKIPPED")
                continue
            print(nickname)
            annotation, score = ANNOTATIONSCOREDUPLES[nickname]
            annotation = osp.join(augnet_repo, annotation)
            score = osp.join(augnet_repo, score)
            stats = store_labeled_pitch_array_and_label_tsv(
                nickname,
                score,
                annotation,
                datasetDir,
                split,
                assembled_dir,
                include_metadata,
            )
            statsrecords.append(stats)
            jointdf = pd.DataFrame.from_records(statsrecords)
            jointdf.to_csv(dataset_summary_path, sep="\t", index=False)
    return jointdf


# %%
output_dir = osp.join("..", "..", "pitch_arrays", "AN")
assembled_scores = osp.join("..", "AN_mscx")
generateEventsDataset(augnet_repo=REPO_PATH, tsvDir=output_dir, assembled_dir=assembled_scores, reset=True)

# %%
