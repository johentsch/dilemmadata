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
#     display_name: ms3
#     language: python
#     name: ms3
# ---

# %%
import re
import ms3
import numpy as np


def absolute_localkey_to_numeral(localkey_abs: str, globalkey: str):
    lc_is_minor = localkey_abs.islower()
    gc_is_minor = globalkey.islower()
    lc_fifths = ms3.name2fifths(localkey_abs)
    gc_fifths = ms3.name2fifths(globalkey)
    lc_roman = ms3.fifths2rn(lc_fifths - gc_fifths, minor=gc_is_minor, auto_key=False)
    return lc_roman.lower() if lc_is_minor else lc_roman


# %%
dataset = ms3.Corpus("../assembled" , only_metadata_pieces=False, file_re="wtc",)
dataset

# %%
dataset.parse(parallel=False)

# %%
def replace_abc_dom9(label, localkey):
    
    def ninth_in_parentheses(match):
        nonlocal is_minor
        numeral = match.group(1)
        ninth = match.group(2)
        accidentals = match.group(3)
        modifiers = match.group(4)
        relative_key = match.group(5)
        if accidentals and "b" in accidentals:
            if relative_key: # we can correct this tacitly for relative minor keys (only those are captured by the rgx)
                ninth = "9"
            elif is_minor:   # for "normal" dominants, this is a mistake 
                raise ValueError(f"b9 does not make sense in a minor scale: {match.group(0)}")
            else: # in major, this is fine
                pass
            
            
        result = f"{numeral}({ninth}{modifiers})" if modifiers else f"{numeral}(+{ninth})"
        if relative_key:
            result += "/" + relative_key
        return result
        
    is_minor = localkey.islower()
    regex = r"(V|iv|iii|ii)((#+|b+)?9)(?:\((\S+)\))?(?:\/#*b*(vii|vi|v|iv|iii|ii|i))?"
    new_label = re.sub(regex, ninth_in_parentheses, label)
    if new_label != label:
        print(f"{label} => {new_label}")
    return new_label


def replace_tavern_dom9(label, localkey):
    
    def M9_in_parentheses(match):
        nonlocal is_minor
        dominant = match.group(1)
        return f"{dominant}(+#9)" if is_minor else f"{dominant}(+9)"
    
    def m9_in_parentheses(match):
        nonlocal is_minor
        dominant = match.group(1)
        return f"{dominant}(+9)" if is_minor else f"{dominant}(+b9)"
        
    is_minor = localkey.islower()
    M9_regex = r"(V(?:7|43|65))M9"
    new_label = re.sub(M9_regex, M9_in_parentheses, label)
    m9_regex = r"(V(?:7|43|65))b9"
    new_label = re.sub(m9_regex, m9_in_parentheses, new_label)
    if new_label != label:
        print(f"{label} => {new_label}")
    return new_label

for piece_id, piece in dataset.iter_pieces():
    print(piece_id, end=" -> ")
    _, labels_df = piece.get_parsed_tsv("labels")
    labels_df = labels_df.copy()
    labels_df.label = (
        labels_df.label
        .str.replace(r"(\d)/(\d)", r"\1\2", regex=True) # imported ABC's inversion digits are separated by /
        .str.replace("IV/64", "IV64", regex=False)  
        .str.replace("maj", "M", regex=False)           
        .str.replace(r"Cad64", r"V(64)", regex=False)    
        .str.replace("Vd", "V", regex=False)            # used in TAVERN for V2 but interpreted as V in rntxt
        .str.replace(r"(oø|ø)", "%", regex=True)
        .str.replace("N", "bII", regex=False)
        .str.replace("^bII#7$", "biiM7", regex=True)
        .str.replace("^bII6#5$", "biiM65", regex=True)
        .str.replace("^bII4#3$", "biiM43", regex=True)
        .str.replace("^IV6b5$", "IV65", regex=True)
        .str.replace("/viio", "/vii", regex=False) 
        .str.replace(r"^(V|I|v)4$", r"\1(4)", regex=True)
        .str.replace("^ivb$", "iv", regex=True)
        .str.replace(r"^I6\+$", "I6", regex=True) # winterreise-d-911-16-letzte-hoffnung (should've been augmented)
        .str.replace("^(V|I|i)54$", r"\1(4)", regex=True)       # romantische-gesange-10-ida-aus-ariels-offenbarungen
        .str.replace("^i:$", "i", regex=True)
        .str.replace("^It53$", "#vo(b3)/V", regex=True)
        .str.replace(r"^It53\[add4\]$", "#vo(+4b3)/V", regex=True)
        .str.replace("^viio/42$", "viio42", regex=True) # this is meant as viiø42 but misinterpreted as viio42
        .str.replace(r"^(VI|iv|i)b7$", r"\g<1>7", regex=True)      # wir-openscore-liedercorpus-chaminade-amoroso    
        .str.replace("-VI", "VI", regex=False)          # tavern-beethoven-woo-75-a, m. 461
        .str.replace(r"\[\S+\]", "", regex=True)         # alternative labels in TAVERN
        .str.replace("I7+6", "I+M65", regex=False)      # tavern-beethoven-woo-65-a, m. 122
        .str.replace("Vi", "V7", regex=False)           # bps-16-op031-no1-1, m. 84
        .str.replace("V7IV", "V7/IV", regex=False)      # bps-26-op081a-les-adieux-1, m. 66
    ) 
    localkey = ms3.transform(labels_df, absolute_localkey_to_numeral, ["localkey_abs", "globalkey"])
    localkey_changes = (localkey != localkey.shift(1))
    modulations = (localkey.where(localkey_changes) + ".").fillna("")
    modulations.iat[0] = labels_df.iloc[0].globalkey + "."
    labels_df["localkey"] = localkey
    labels_df.label = ms3.transform(labels_df, replace_abc_dom9, ["label", "localkey"])
    labels_df.label = ms3.transform(labels_df, replace_tavern_dom9, ["label", "localkey"])
    labels_df.label = modulations + labels_df.label 
    piece.load_annotation_table_into_score(df=labels_df)
    score_id, score = piece.get_parsed("score")
    detached_labels = score["detached"].df
    incompatible_mask = detached_labels.regex_match.isna()
    if incompatible_mask.any():
        # break
        print("OK with problems")
    else:
        print("OK")
    score.attach_labels(
        key="detached",
        staff=None,
        voice=None,
        harmony_layer=1,
        check_for_clashes=True,
    )
    score.store_score(score.musescore_file)
    
detached_labels[incompatible_mask]

# %%
detached_labels[incompatible_mask].label.str.replace("^It53$", "#vo(b3)/V", regex=True)

# %%
_, notes = piece.get_facet("notes")
notes

# %%
fractions_list = notes.quarterbeats.tolist()
print(len(fractions_list))
fractions_list

# %%
lcm = np.lcm.reduce([fr.denominator for fr in fractions_list])

vals = [int(fr.numerator * lcm / fr.denominator) for fr in fractions_list]
#vals.append(lcm)
vals

# %%
score.attach_labels(
        key="detached",
        staff=None,
        voice=None,
        harmony_layer=1,
        check_for_clashes=True,
    )
score.store_score(score.musescore_file)
score

# %%
score.detach_labels("detached")
score.store_score(score.musescore_file)
score

# %%
