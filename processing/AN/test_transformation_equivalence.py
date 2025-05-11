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

import ms3
import pandas as pd

# %%
# %load_ext autoreload
# %autoreload 2
from AugmentedNet import utils

# %%
DATASET_PATH = ms3.resolve_dir("../events")
print(DATASET_PATH)

# %%
pitch_arrays = {}
for split in os.listdir(DATASET_PATH):
    split_dir = os.path.join(DATASET_PATH, split)
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
concat = pd.concat(pitch_arrays)
concat

# %%
concat[concat.a_simpleNumeral.isna()].index.get_level_values(0).unique()

# %%
augnet_numeral_counts = concat.a_romanNumeral.value_counts()
augnet_numeral_counts.iloc[:100]

# %%
simpleNumeral_dirty = concat.a_romanNumeral.str.split("/", expand=True).iloc[:, 0]
simpleNumeral_dirty = simpleNumeral_dirty.str.replace("-", "b")
simpleNumeral_dirty = simpleNumeral_dirty.str.replace("ø", "%")
simpleNumeral_dirty = simpleNumeral_dirty.str.replace("54", "")
simpleNumeral_dirty = simpleNumeral_dirty.str.replace("NI", "N")
simpleNumeral_dirty = simpleNumeral_dirty.replace("Vd", "V7")
simpleNumeral_dirty = simpleNumeral_dirty.replace("viio3", "viio")
simpleNumeral_counts = (
    simpleNumeral_dirty.value_counts()
    .to_frame("simpleNumeral")
    .reset_index()
    .rename(columns=dict(index="label"))
)
simpleNumeral_vocab = simpleNumeral_counts.index.tolist()
simpleNumeral_counts

# %%

# matches = simpleNumeral_counts.label.str.match(utils.SIMPLE_RN_REGEX)
# simpleNumeral_counts[~matches]
simpleNumeral_components = simpleNumeral_dirty.str.extract(
    utils.SIMPLE_RN_REGEX
).fillna("")
simpleNumeral_components

# %%
simpleNumeral_clean = simpleNumeral_components.loc[:, "acc":].sum(axis=1)
simpleNumeral_clean = simpleNumeral_clean.where(simpleNumeral_clean != "", "none")
simpleNumeral_clean_counts = simpleNumeral_clean.value_counts()
simpleNumeral_clean_counts

# %%
utils.print_rn_stats(simpleNumeral_clean)

# %%
simpleNumeral_clean_counts.index.tolist()

# %%
top75 = augnet_numeral_counts.iloc[:75].index.tolist()
for rn in top75:
    print('"{0}",'.format(rn))

# %%

dlc_labels = ms3.load_tsv(
    "/home/laser/Documents/Linz/DLC_version_comparison/distant_listening_corpus_v3.1/"
    "distant_listening_corpus.expanded.tsv"
)
dlc_labels

# %%

dlc_labels[dlc_labels.chord.isna()]

# %%
dlc_measures = ms3.load_tsv(
    "/home/laser/Documents/Linz/DLC_version_comparison/distant_listening_corpus_v3.1/"
    "distant_listening_corpus.measures.tsv"
)
dlc_measures.keysig.value_counts(dropna=False)

# %%
arr = pitch_arrays["wirwtc-bach-wtc-i-14"]
utils.create_specs(arr)

# %%
concat.a_quality.value_counts().sort_index()

# %%
DLC = ms3.load_tsv(
    "/home/laser/Documents/Linz/DLC_version_comparison/distant_listening_corpus_v3.1/"
    "distant_listening_corpus.expanded.tsv"
)
DLC.chord_type.value_counts(dropna=False)

# %%
DLC_CHORD_TYPE_MAPPING = {
    "M": "major triad",
    "m": "minor triad",
    "o": "diminished triad",
    "+": "augmented triad",
    "+7": "augmented triad",  # actually "augmented seventh chord"
    "+M7": "augmented triad",  # actually "augmented major tetrachord"
    "mm7": "minor seventh chord",
    "MM7": "major seventh chord",
    "Mm7": "dominant seventh chord",
    "incomplete dominant-seventh chord": "incomplete dominant-seventh chord",  # currently not available in DLC
    "o7": "diminished seventh chord",
    "%7": "half-diminished seventh chord",
    "It": "Italian augmented sixth chord",
    "Ger": "German augmented sixth chord",
    "Fr": "French augmented sixth chord",
    "mM7": "minor-augmented tetrachord",
    pd.NA: "None",
}
DLC.chord_type.map(DLC_CHORD_TYPE_MAPPING).value_counts(dropna=False)

# %%
concat[concat.a_quality == "minor-augmented tetrachord"]

# %% [markdown]
# ### Making sure roman_numeral2scale_degree() computes music21-equivalent scale degrees for all numerals

# %%

concat["numeral"] = concat.a_romanNumeral.str.extract(utils.ROOT_RN_REGEX)

# roman_numeral2scale_degree() (when key_is_minor=True) determines chord quality purely based on the numeral's third,
# as expressed by it being in lowercase (m3) or uppercase (M3). Therefore, the numerals of augmented chords written in
# lowercase need to be converted to upper case. A regEx is used to convert only the characters i and v.
upper_case_numeral = concat.numeral.str.replace(
    "([iv])", lambda match: match.group(1).upper(), regex=True
)
concat.numeral = concat.numeral.where(
    concat.a_quality != "augmented triad", upper_case_numeral
)
concat["tonicizedkey_is_minor"] = concat.a_tonicizedKey.str.islower()
concat["degree1"] = ms3.transform(
    concat,
    utils.roman_numeral2scale_degree,
    ["numeral", "tonicizedkey_is_minor"],
    flat_character="-",
)

# %%
not_matching_mask = (concat.a_degree1 != concat.degree1.fillna("")) & (
    concat.a_degree2 != "None"
)
concat[not_matching_mask]
