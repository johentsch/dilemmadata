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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# DON'T EVEN TRY TO RUN THIS ON WINDOWS

import os
import re
from typing import Dict

import git
import ms3
import pandas as pd

from AugmentedNet import utils
from AugmentedNet.common import ANNOTATIONSCOREDUPLES, DATASPLITS


def resolve_dir(d):
    """Resolves '~' to HOME directory and turns ``d`` into an absolute path."""
    if d is None:
        return None
    d = str(d)
    if "~" in d:
        return os.path.expanduser(d)
    return os.path.abspath(d)


REPO_PATH = resolve_dir("..")
DATASET = "events"
AUGMENTEDNET_REPO = git.Repo(REPO_PATH)
AUGMENTEDNET_VERSION = "v1.0.0"
aug_ver = AUGMENTEDNET_VERSION.replace(".", "")
id_col, split_col = f"id_{aug_ver}", f"split_{aug_ver}"
REGENERATE = False
print(REPO_PATH)


# %%
def add_safely(dictionary, key, pair):
    if key in dictionary:
        left, right = pair
        existing_left, existing_right = dictionary[key]
        if isinstance(existing_left, str):
            new_pair = ((existing_left, left), (existing_right, right))
        else:
            new_pair = (existing_left + (left,), existing_right + (right,))
        dictionary[key] = new_pair
    else:
        dictionary[key] = pair


def assemble_ids_and_splits(datasplits, annotationscoreduples):
    path2name_and_split = {}

    n = 0
    for split, files in datasplits.items():
        for nickname in files:
            annotations_path, score_path = annotationscoreduples[nickname]
            file_info = (nickname, split)
            add_safely(path2name_and_split, annotations_path, file_info)
            n += 1
            add_safely(path2name_and_split, score_path, file_info)
            n += 1
    return path2name_and_split, n


path2name_and_split, n_files = assemble_ids_and_splits(
    DATASPLITS, ANNOTATIONSCOREDUPLES
)
print(
    f"{n_files} uses of {len(path2name_and_split)} files overall (some scores are used multiple times)."
)
# assert len(path2name_and_split) == i * 2, f"dict length {len(path2name_and_split)} != {i * 2} ({i} * 2)"

# %%
SUBMODULE_REPOS: Dict[str, git.Repo] = {
    sm.name: sm.module() for sm in AUGMENTEDNET_REPO.submodules
}
SUBMODULE_VERSIONS = {
    name: sm_repo.git.describe(tags=True, always=True)
    for name, sm_repo in SUBMODULE_REPOS.items()
}
SUBMODULE_VERSIONS

# %%
REPO_URLS = {
    "AugmentedNet": "https://github.com/napulen/AugmentedNet",
    "TAVERN": "https://github.com/jcdevaney/TAVERN",
    "ABC": "https://github.com/DCMLab/ABC",
    "haydn_op20_harm": "https://github.com/napulen/haydn_op20_harm",
    "When-in-Rome": "https://github.com/MarkGotham/When-in-Rome",
    "music21_corpus": "https://github.com/cuthbertLab/music21",
    "functional-harmony-micchi": "https://github.com/napulen/functional-harmony-micchi",
}

# %%
EXCLUDED_EXTENSIONS = (".h5", ".jl", "krn~", ".md", ".pdf", ".py", ".sh", ".swp")
# (".cfg", ".css", ".csv", ".h5", ".html", ".in", ".ipynb", ".jl", ".js", ".md", ".pdf", ".png", ".py", ".rst", ".sh",
# ".tsv", ".yml")
EXCLUDED_NAME_COMPONENTS = ("feedback", "license", "slices", "template", "requirements")
PRINT_SYMBOLS = dict(validation="/", training="|", test="\\")
PATH_FILTERS = {
    # for rel_paths matching a key, go only through the subdirectories in the corresponding list
    os.path.join("rawdata", "When-in-Rome"): ["Corpus"],
    os.path.join("rawdata", "music21_corpus"): ["music21"],
    os.path.join("rawdata", "music21_corpus", "music21"): ["corpus"],
    os.path.join("rawdata", "music21_corpus", "music21", "corpus"): [
        "bach",
        "monteverdi",
    ],
}
SUBCORPUS_POSITION = {
    "AugmentedNet": 2,  # rawdata/corrections/ABC
    "TAVERN": 0,  # TAVERN/Beethoven
    "ABC": None,
    "haydn_op20_harm": None,
    "When-in-Rome": 1,  # When-in-Rome/Corpus/Early_Choral
    "music21_corpus": 2,  # music21/corpus/bach
    "functional-harmony-micchi": 1,  # data/19th_Century_Songs
}


def get_commit_where_file_last_changed(repo: git.Repo, paths=str):
    try:
        return next(repo.iter_commits(paths=paths))
    except StopIteration as e:
        raise StopIteration(f"{repo!r} does not have any commits for {paths}") from e


def create_data_overview(rawdata_path, path2name_and_split, augnet_version):
    data = []
    for data_dir in os.listdir(rawdata_path):
        if data_dir == "TAVERN":
            continue  # the original files are not actually used and there are many, many, many
        print(f"\n{data_dir}")
        data_dir_path = os.path.join(rawdata_path, data_dir)
        if data_dir in SUBMODULE_VERSIONS:
            current_repo_name = data_dir
            git_path_base = data_dir_path
        else:
            current_repo_name = "AugmentedNet"
            git_path_base = REPO_PATH
        current_repo = SUBMODULE_REPOS.get(data_dir, AUGMENTEDNET_REPO)
        current_repo_version = SUBMODULE_VERSIONS.get(data_dir, AUGMENTEDNET_VERSION)
        current_repo_url = REPO_URLS.get(current_repo_name).strip("/")
        subcorpus_position = SUBCORPUS_POSITION.get(current_repo_name)
        for path, subdirs, files in os.walk(data_dir_path):
            rel_path = os.path.relpath(path, REPO_PATH)
            if rel_path in PATH_FILTERS:
                subdirs[:] = PATH_FILTERS[rel_path]
                continue
            for file in files:
                fname, fext = os.path.splitext(file)
                if not fext or fext in EXCLUDED_EXTENSIONS:
                    print(".", end="")
                    continue
                fname_lower = fname.lower()
                if any(comp in fname_lower for comp in EXCLUDED_NAME_COMPONENTS):
                    print(".", end="")
                    continue
                filepath = os.path.join(rel_path, file)
                folder_name = os.path.basename(rel_path)
                git_filepath = os.path.relpath(os.path.join(path, file), git_path_base)
                subcorpus = None
                if subcorpus_position is not None:
                    split_git_path = git_filepath.split(os.sep)
                    try:
                        subcorpus = split_git_path[subcorpus_position]
                    except Exception:
                        pass
                file_last_changed_commit = get_commit_where_file_last_changed(
                    current_repo, paths=git_filepath
                )
                file_last_changed_commit_sha = file_last_changed_commit.hexsha
                file_last_changed_commit_version = current_repo.git.describe(
                    file_last_changed_commit_sha, tags=True, always=True
                )
                file_change_commit_url = f"{current_repo_url}/blob/{file_last_changed_commit_version}/{git_filepath}"
                aug_ver = augnet_version.replace(".", "")
                info_dict = dict(
                    dataset=data_dir,
                    subcorpus=subcorpus,
                    file=file,
                    fname=fname,
                    extension=fext[1:],
                )
                if filepath in path2name_and_split:
                    nickname, split = path2name_and_split[filepath]
                    which_set = split if isinstance(split, str) else split[0]
                    print_symbol = PRINT_SYMBOLS.get(which_set)
                else:
                    print_symbol = ":"
                    nickname, split = None, None
                info_dict.update(
                    {
                        f"id_{aug_ver}": nickname,
                        f"split_{aug_ver}": split,
                        f"last_modified_{aug_ver}": file_last_changed_commit_version,
                        f"file_change_commit_url_{aug_ver}": file_change_commit_url,
                        "repository": current_repo_name,
                        f"repo_version_{aug_ver}": current_repo_version,
                        "folder": folder_name,
                        "folderpath": rel_path,
                        "filepath": filepath,
                    }
                )
                data.append(info_dict)
                print(print_symbol, end="")
    return pd.DataFrame.from_records(data).sort_values(
        ["dataset", "subcorpus", "file", "filepath"]
    )


rawdata_path = os.path.join(REPO_PATH, "rawdata")
tsv_path = "../../corpora/AugmentedNet/augnet_rawdata_v100.tsv"
if REGENERATE:
    df = create_data_overview(
        rawdata_path,
        path2name_and_split=path2name_and_split,
        augnet_version=AUGMENTEDNET_VERSION,
    )
    df.to_csv(tsv_path, sep="\t", index=False)
else:
    dtype = {id_col: object, split_col: object}
    df = pd.read_csv(tsv_path, sep="\t", dtype=dtype)
    tuple_mask = df[id_col].str.startswith("(").fillna(False)
    df.loc[tuple_mask, [id_col, split_col]] = df.loc[
        tuple_mask, [id_col, split_col]
    ].applymap(utils.safe_literal_eval)
df

# %%
attributed_filepaths = df[split_col].notna().sum()
assert attributed_filepaths == len(path2name_and_split), (
    f"Not all of the {len(path2name_and_split)} used files have been attributed in the Dataframe (containing "
    f"{attributed_filepaths}), probably due to exclusion criteria."
)

# %% [markdown]
# # Joint overview
# ## AugmentedNet part

# %%

augnet = df[df[split_col].notna()].copy()
value_type = augnet[id_col].map(type)
tuple_mask = value_type == tuple
exploded_tuples = augnet[tuple_mask].explode([id_col, split_col])
augnet = pd.concat([augnet[~tuple_mask], exploded_tuples])
# corpus_col = (augnet.subcorpus
#               .fillna(augnet.dataset)
#               .replace(
#                     {
#                         "Beethoven_4tets": "ABC",
#                         "Early_Choral": "bach_chorales",
#                         "Etudes_and_Preludes": "WTC",
#                         "OpenScore-LiederCorpus": "lieder_corpus",
#                         "Piano_Sonatas": "BPS",
#                         "Variations_and_Grounds": "Tavern"
#                     }
#                 )
#               .rename("corpus"))
corpus_col = augnet[id_col].str.split("-", expand=True)[0].rename("corpus")
augnet = pd.concat([corpus_col, augnet], axis=1)
augnet = augnet.sort_values(
    by=["corpus", split_col, id_col], ascending=True
).reset_index(drop=True)
augnet

# %%
augnet.extension.value_counts()

# %%
is_analysis = augnet.extension == "txt"
select_columns = [
    "corpus",
    split_col,
    id_col,
    "filepath",
    f"file_change_commit_url_{aug_ver}",
]
augnet_summary = (
    pd.merge(
        left=augnet.loc[~is_analysis, select_columns],
        right=augnet.loc[is_analysis, select_columns],
        on=["corpus", id_col, split_col],
        suffixes=("_score", "_annotation"),
    )
    .set_index(id_col, drop=False)
    .astype("string")
)
augnet_summary

# %%
augnet_summary.to_csv("../augnet_summary_v100.tsv", sep="\t", index=False)

# %% [markdown]
# ## DLC part

# %%
DLC_PATH = ms3.resolve_dir(
    "~/distant_listening_corpus"
)  # needs to be checked out at the right path (currenty "pitch_arrays")
dlc = ms3.Parse(DLC_PATH)
dlc.view.include("facet", "expanded")
dlc.parse_tsv()
dlc_annotations = dlc.get_facet("expanded")
dlc_annotations


# %%
def column_is_present_and_not_empty(df: pd.DataFrame, col_name: str) -> bool:
    if col_name not in df.columns:
        return False
    return df[col_name].notna().any()


def detect_label_types(df: pd.DataFrame):
    """Takes a harmony TSV and looks up which types of annotation labels are present."""
    return pd.Series(
        dict(
            has_chords=column_is_present_and_not_empty(df, "chord"),
            has_cadence=column_is_present_and_not_empty(df, "cadence"),
            has_phrase=column_is_present_and_not_empty(df, "phraseend"),
            has_pedal=column_is_present_and_not_empty(df, "pedal"),
        )
    )


label_types = dlc_annotations.groupby(["corpus", "piece"]).apply(detect_label_types)
label_types.has_pedal = True
label_types

# %%
metadata_path = os.path.join(
    DLC_PATH, "processing", "distant_listening_corpus.metadata.tsv"
)
last_modified_url = ms3.load_tsv(
    metadata_path,
    index_col=["corpus", "piece"],
    usecols=["corpus", "piece", "last_modified_url", "rel_path"],
)
last_modified_url

# %%
dlc_summary = last_modified_url.join(label_types, how="right").astype(
    {
        "last_modified_url": "string",
        "rel_path": "string",
        "has_chords": "boolean",
        "has_cadence": "boolean",
        "has_phrase": "boolean",
        "has_pedal": "boolean",
    }
)
dlc_summary.to_csv("../dlc_summary.tsv", sep="\t")
dlc_summary

# %%
dlc_summary.iloc[:, 2:].sum()

# %% [markdown]
# **Check which piece comes without phrase annotations.**

# %%
dlc_summary[~dlc_summary.has_phrase]

# %% [markdown]
# ## Joining them together
# ### ABC

# %%
dlc_ids = dlc_summary.index.to_frame()
abc_ids_dlc = dlc_ids.loc[["ABC"]]
abc_naming_dlc = r"""
n(?P<quartet>\d{2})
op(?P<op>\d{2,3})
(?:-(?P<no>\d))?
_(?P<mvt>\d{2})
"""
abc_names_dlc = abc_ids_dlc.piece.str.extract(abc_naming_dlc, flags=re.VERBOSE).astype(
    "Int64"
)

abc_naming_augnet = r"""
abc-op(?P<op>\d+)
(?:-no(?P<no>\d))?
-(?P<mvt>\d)
"""
abc_ids_augnet = augnet_summary.loc[augnet_summary.corpus == "abc", id_col]
abc_names_augnet = abc_ids_augnet.str.extract(
    abc_naming_augnet, flags=re.VERBOSE
).astype("Int64")

merged_abc_ids = pd.merge(
    left=abc_names_augnet.reset_index(),
    right=abc_names_dlc.reset_index(),
    on=["op", "no", "mvt"],
    how="left",
)
dlc_index2augnet_ids = (
    merged_abc_ids.set_index(["corpus", "piece"])[id_col]
    .astype("string")
    .reindex(dlc_summary.index)
)
dlc_index2augnet_ids

# %% [markdown]
# ### BPS

# %%
bps_ids_augnet = augnet_summary.loc[augnet_summary.corpus == "bps", id_col]
bps_number_augnet = (
    bps_ids_augnet.str.split("-", expand=True).iloc[:, 1].rename("piece") + "-1"
)
for augnet_idx, piece in bps_number_augnet.items():
    if (dlc_idx := ("beethoven_piano_sonatas", piece)) in dlc_index2augnet_ids.index:
        dlc_index2augnet_ids.loc[dlc_idx] = augnet_idx
dlc_index2augnet_ids

# %% [markdown]
# ### Monteverdi Madrigals

# %%
# keys from: dlc_ids.loc[["monteverdi_madrigals"]].index.to_list()
# values from: augnet_summary.loc[augnet_summary[id_col].str.startswith("wir-monteverdi-madrigals"), id_col]
# .sort_values().tolist()
schumann_mapping = {
    ("monteverdi_madrigals", "3-09"): "wir-monteverdi-madrigals-book-3-11",
    # ('monteverdi_madrigals', '4-19'): actually matches 'wir-monteverdi-madrigals-book-4-20', not included in AugNet
    ("monteverdi_madrigals", "5-04a"): "wir-monteverdi-madrigals-book-5-4",
    # ('monteverdi_madrigals', '5-04b') missing in the DLC, would match 'wir-monteverdi-madrigals-book-5-5'
    ("monteverdi_madrigals", "5-04d"): "wir-monteverdi-madrigals-book-5-7",
    ("monteverdi_madrigals", "5-04e"): "wir-monteverdi-madrigals-book-5-8",
}
for dlc_idx, augnet_idx in schumann_mapping.items():
    dlc_index2augnet_ids.loc[dlc_idx] = augnet_idx

# %% [markdown]
# ### Schubert Winterreise

# %%
winterreise_indexer = augnet_summary[id_col].str.startswith(
    "wir-openscore-liedercorpus-schubert-winterreise"
)
for i, augnet_idx in enumerate(
    augnet_summary.loc[winterreise_indexer, id_col].sort_index(), 1
):
    dlc_idx = ("schubert_winterreise", f"{i:02}")
    dlc_index2augnet_ids.loc[dlc_idx] = augnet_idx

# %% [markdown]
# ### C. Schumann, op. 13

# %%
# keys from: dlc_ids.loc[["c_schumann_lieder"]].index.to_list()
# values from: augnet_summary.loc[augnet_summary[id_col]
#               .str.startswith("wir-openscore-liedercorpus-schumann-6-lieder-op-13"), id_col].tolist()
schumann_mapping = {
    (
        "c_schumann_lieder",
        "op13no1 Ich stand in dunklen Traumen",
    ): "wir-openscore-liedercorpus-schumann-6-lieder-op-13-1-ich-stand-in-dunklen-traumen",
    (
        "c_schumann_lieder",
        "op13no2 Sie liebten sich beide",
    ): "wir-openscore-liedercorpus-schumann-6-lieder-op-13-2-sie-liebten-sich-beide",
    (
        "c_schumann_lieder",
        "op13no3 Liebeszauber",
    ): "wir-openscore-liedercorpus-schumann-6-lieder-op-13-3-liebeszauber",
    (
        "c_schumann_lieder",
        "op13no6 Die stille Lotosblume",
    ): "wir-openscore-liedercorpus-schumann-6-lieder-op-13-6-die-stille-lotosblume",
}
for dlc_idx, augnet_idx in schumann_mapping.items():
    dlc_index2augnet_ids.loc[dlc_idx] = augnet_idx

# %% [markdown]
# ### The Merge

# %%
dlc_summary[id_col] = dlc_index2augnet_ids
merged = pd.merge(
    dlc_summary.reset_index(),
    augnet_summary.reset_index(drop=True),
    on=id_col,
    how="outer",
    suffixes=("_dlc", "_augnet"),
)
augnet_only = merged.corpus_dlc.isna()
merged = pd.concat(
    [
        merged[augnet_only].sort_values(["corpus_augnet", split_col, id_col]),
        merged[~augnet_only].sort_values(["corpus_dlc", "piece"]),
    ]
)
merged.to_csv("../merged_summary.tsv", sep="\t", index=False)
merged

# %%
augnet_only = merged.corpus_dlc.isna()
merged_sorted = pd.concat(
    [
        merged[augnet_only].sort_values(["corpus_augnet", split_col, id_col]),
        merged[~augnet_only].sort_values(["corpus_dlc", "piece"]),
    ]
).reset_index(drop=True)

# %%
merged_sorted.has_chords = merged_sorted.has_chords.fillna(True)
merged_sorted.loc[:, ["has_cadence", "has_phrase", "has_pedal"]] = merged_sorted.loc[
    :, ["has_cadence", "has_phrase", "has_pedal"]
].fillna(False)
char_cols = pd.DataFrame(
    {
        col: pd.Series(col, index=merged_sorted.index).where(merged_sorted[mask], "")
        for col, mask in zip(
            ("H", "C", "P"), ("has_chords", "has_cadence", "has_phrase")
        )
    }
)
merged_sorted["label_combination"] = char_cols.sum(axis=1)
merged_sorted.to_csv("../merged_summary.tsv", sep="\t", index=False)
merged_sorted

# %%
label_combination = pd.Series(
    list(
        merged_sorted[["has_chords", "has_cadence", "has_phrase"]].itertuples(
            index=False, name=None
        )
    ),
    index=merged_sorted.index,
)
label_combination.value_counts()

# %% [markdown]
# ## Exclude DLC that is part of Augnet test set

# %%
excluded_mask = merged_sorted.corpus_dlc.notna() & (merged_sorted[split_col] == "test")
excluded_dlc = merged_sorted[excluded_mask]
excluded_ids = excluded_dlc.corpus_dlc + "_" + excluded_dlc.piece
excluded_ids.tolist()

# %%
