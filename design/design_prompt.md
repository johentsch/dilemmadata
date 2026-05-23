# design_prompt.md — Conference Poster Brief for **dilemmadata**

This is a self-contained briefing for Claude Design. The goal is a conference poster (DLfM '25 / MEC 2026 / similar venue) that communicates what dilemmadata is, why it exists, how it works, and the conceptual provocation behind it. The poster should be readable from across a room (banner numbers + one strong visual) and rewarding up close (worked example, regex, mappings).

---

## 0. Abstract

In recent years, there has been growing effort to annotate and collect large-scale corpora of Roman numeral analyses in support of data-driven studies in tonal harmony.
We introduce _dilemmadata_, the first resource to reconcile two major collections — the AugmentedNet Dataset (AN) and the Distant Listening Corpus (DLC) — making them interoperable through a shared note-wise TSV schema.
The reconciliation confronts four families of dilemmata: *annotation-standard* (the two encode the same musical fact differently — vocabulary size, syntax, conventions for chord extensions, inventory of special chord functions), *representational* (what counts as a row, and which information survives the conversion), *toolchain* (incompatible Python ecosystems built around `music21` vs. `ms3`+`dimcat`), and *curatorial* (which pieces to include, exclude, or retain twice).
We resolve each by deliberately transforming, augmenting, and omitting information, formalising the mismatches, preserving musical semantics, and flagging transformations that may subtly affect annotation fidelity.
Consistency checks and qualitative inspections offer a preliminary assessment of post-conversion validity and a basis for critiquing the theoretical assumptions embedded in each original standard.
After removing duplicates and merging the two collections, the resulting _dilemmadata_ — 1,621 pieces and ~2.8 M note-wise annotations — is the largest homogeneous Roman-numeral corpus currently available, albeit far from perfect.
Crucially, we retain 84 pieces common to both corpora under each of their original analyses, yielding a shared reference set in which two equally legitimate analytical traditions can be compared note-for-note over identical musical material.
Released at <https://doi.org/10.5281/zenodo.19661223>, _dilemmadata_ supports interoperability, comparative harmonization modeling, and future refinement of Roman-numeral encoding standards.

### 0.1 The dilemmata (why the name)

Merging heterogeneous Roman-numeral datasets means facing many dilemmata — hence the name. Four families, enumerated below. Entries marked ⭐ are the ones most likely to anchor a poster panel.

- **Annotation-standard dilemmata** — places where RomanText (AN) and DCML (DLC) encode the *same musical fact* differently, forcing a unification choice:
  - ⭐ **Cadential 6/4.** RomanText writes `I64` (or `Cad64`); DCML writes `V(64)` (dominant with a 6–5 / 4–3 double suspension). Same notes, same function, *categorically different chord root.* Both are rewritten to a shared symbol `Cad`. It is the seventh-most-frequent harmony in the joined output — not a corner case.
  - **Minor-mode VI / VII.** DCML defaults to natural minor (so `VII` and `viio` share a root); RomanText infers mode from chord quality. Handled in `add_simpleNumeral_columns`.
  - **Chord-extension vocabulary.** DCML's `Mm7`, `%7`, `Ger`, `It`, `Fr`, `N` mapped onto music21's verbal labels (`dominant seventh chord`, `half-diminished seventh chord`, `German augmented sixth chord`, …) via `DLC_CHORD_TYPE_MAPPING`.
  - **Inversion notation.** Figured-bass strings (`6`, `65`, `64`, `43`, `42`, `2`) collapsed to integers 0–3 via `FIGBASS2INVERSION`.
  - **Encoding paradigm.** Stand-off `.rntxt` text paired with heterogeneous scores (AN) vs. annotations embedded inside `.mscx` files attached to specific notes (DLC). Two parser ecosystems (`music21` vs. `ms3`+`dimcat`) follow from this split.

- **Representational dilemmata** — what counts as a row, and how much information survives the conversion:
  - ⭐ **Salami slice vs. note event.** Upstream AugmentedNet emits one row per fixed 16th-note "Salami slice" (~100 K rows); the dilemmadata fork bypasses `music21.chordify` and emits one row per *note event* (~750 K rows). A methodological shift, not a cosmetic one — it enables true note-wise (rather than slice-wise) learning.
  - **Transform / augment / omit.** The conversion mapping makes choices in all three directions; the abstract names them deliberately because each direction silently affects what downstream models can learn. Validity masks flag the cases where conversion is lossy.
  - **Structural vs. ornamental notes.** The two standards carry different implicit views about which notes "belong" to the harmony and which are decoration. The schema preserves the source verdict where it can and flags where it cannot.

- **Toolchain dilemmata** — the two corpora demand incompatible Python stacks, and the stacks themselves drift:
  - **Pipeline divergence.** AN parses with `music21` via `AugmentedNet.joint_parser`; DLC parses with `ms3` + `dimcat`. Schema alignment had to be triangulated between the two pipelines *and* AnalysisGNN's task vocabularies, with no single authority to defer to.
  - **The `music21` version dilemma.** AN v1.0.0 was authored against `music21` v6.7.0; current `dimcat` pulls v9.5.0, which is stricter (e.g., refuses `b3` in 3/8 without the explicit "slow 3/8" syntax). The fork patches v1.0.0 minimally rather than tracking the latest When-in-Rome — a deliberate, self-aware band-aid.

- **Curation dilemmata** — which pieces to include, exclude, and double-count:
  - ⭐ **The 84 overlapping pieces.** 99 pieces appear in both source corpora; 15 are removed because they live in AN's test split, but the remaining 84 are *deliberately retained under both annotations*. The corpus therefore contains the same music annotated by two equally legitimate analytical traditions, so disagreements between paradigms become *addressable* rather than averaged away.
  - **No objective ground truth.** Roman numeral analysis is interpretive; no metric exists for "correctness" across competing theoretical assumptions. dilemmadata frames this as a feature (study the disagreement) rather than a bug (vote and move on).
  - **"AugmentedNet ground truth" is already a consensus.** AN itself aggregates six upstream corpora (ABC, BPS, HaydnSun, TAVERN, WiR, WTC) of mixed manual and automated provenance, so the apparent asymmetry of "AN vs. DLC" is itself smoothed-over heterogeneity.
  - **The frozen public test set.** Choosing which DLC pieces to set aside (~20 % per subcorpus) had to honour an explicit `excluded_pieces` list and avoid any AN-test overlap. The split is fixed so future models trained on dilemmadata can be compared apples-to-apples.
  - **v1.0.0 vs. v1.9.1 benchmarking.** Comparing a new model against AugmentedNet's published performance forces a choice: train and evaluate on **v1.0.0** (matching the numbers reported in Nápoles López et al. 2021), or on **v1.9.1** (broader coverage, additional When-in-Rome pieces) and re-run AugmentedNet locally to produce a comparable baseline. dilemmadata is built from v1.9.1; downstream users inherit the trade-off.

## 1. One-line pitch

> **dilemmadata** is the largest homogeneous Roman-numeral corpus to date (1,621 pieces, ~2.8 M note-wise annotations) — built by reconciling two incompatible annotation traditions, and deliberately retaining a shared subset of pieces annotated under *both* paradigms as a catalyst for moving beyond ground-truth assumptions in computational harmony.

The name itself is a wink: the project is *data*, but its production was a sequence of *dilemmas* (enumerated in §0.1).

## 2. Audience and venue framing

- **Primary venue:** Music Encoding Conference (MEC 2026).
- **Audience:** Computational musicologists, music-encoding researchers, MIR / ML practitioners working on harmony / Roman numeral analysis, data-curation specialists.
- **They already know:** that Roman numeral analysis exists, that corpora matter for training models.
- **They probably don't know:** the practical scale of incompatibility between RomanText and DCML annotation conventions, the existence of a unified note-wise table for both, the size of dilemmadata relative to prior corpora.
- **Tone:** technically precise but provocative. The word "dilemma" is the hook — lean into it. The poster should make people argue at the coffee break.

## 3. Headline numbers (use as banner / hero stats)

| Metric | Value                                    |
| --- |------------------------------------------|
| Pieces, total | **1,621**                                |
|  — from AugmentedNet (AN **v1.9.1**) | 353                                      |
|  — from Distant Listening Corpus (DLC **v3.1**) | 1,268                                    |
| Note-wise Roman numeral annotations | **~2.8 M**                               |
| AN-side note rows (after `chordify` bypass) | ~750,000 (up from ~100K Salami slices)   |
| AN-side simpleNumeral tokens / types | 88,559 tokens / 132 types (522 `none`)   |
| DLC-side simpleNumeral tokens / types | 318,781 tokens / 185 types (101 `none`)  |
| Overlapping pieces between AND ∩ DLC | **99** (70 of those from ABC)            |
| Overlaps removed (because in AND test split) | 15                                       |
| **Overlaps retained — annotated under both paradigms** | **84** ← *the "dilemma" payload*         |
| DLC sub-corpora | 40+ (Bach → Medtner)                     |
| AND constituent corpora | 6 (ABC, BPS, HaydnSun, TAVERN, WiR, WTC) |
| DLC test split (designed here) | ~20% of DLC pieces                       |
| dilemmadata version | **v1.1**                                 |

## 4. The narrative arc the poster should tell

Three beats, in order:

1. **The problem (left column).** Two large, carefully-curated Roman-numeral corpora exist. They use *the same musical concept* (Roman numeral analysis) but disagree on file formats, parser libraries, syntax for chord extensions, and — most painfully — on what a cadential 6/4 chord *is*. Direct joint use is impractical.
2. **The reconciliation (center, the visual core).** A pipeline that ingests both corpora through their native toolchains (`music21` for AugmentedNet, `ms3`+`dimcat` for DLC), applies surgical transformations and re-encodings, and produces one shared note-wise TSV schema. Visualize: pipeline diagram + a worked Beethoven example showing the two source analyses and the harmonized output side-by-side.
3. **The dilemma (right column).** We don't claim to have produced "the" correct unification — we made *choices*, flagged them with validity masks and provenance hashes, and we kept the 84 overlap pieces under *both* analyses precisely so the community can study where the disagreements live. Future work: `flexohr`, a library to do this for *any* pair of annotation standards.

## 5. Content blocks — what to include

### 5.1 Problem framing (use as a "Disparate encoding paradigms" panel)

- **AugmentedNet (AND):** Roman numeral analyses as stand-off `.rntxt` text files paired with diverse score formats (MusicXML, Humdrum, ABC, kern). Parsed with **music21**. Aggregates six public corpora.
- **Distant Listening Corpus (DLC):** Roman numeral annotations embedded directly *inside* MuseScore (`.mscx`) files, attached to specific notes. Parsed with **ms3** + **dimcat**.
- **Three axes of incompatibility:**
  1. *File format & encoding paradigm* (stand-off text vs. in-score symbols).
  2. *Implicit music-theoretical views* (which note is structural vs. ornamental?).
  3. *Syntax* — chord extensions (7ths, 9ths), special functions (Neapolitan, secondary dominants, cadential 6/4), and minor-mode scale-degree conventions.

### 5.2 The shared schema (the "labeled pitch array")

A note-wise TSV. Each row is one note event. Columns fall into four groups:

- **Position & rhythm:** `onset_div`, `duration_div`, `onset_beat`, `beat_float`, `downbeat`, `is_downbeat`, `ts_beats`, `ts_beat_type`, `mn_playthrough`, `quarterbeats_playthrough`. Uses a common-denominator integer grid (LCM-based) so onsets and durations stay exact across heterogeneous rhythms.
- **Pitch:** `pitch` (MIDI), `tpc` (tonal pitch class, fifths-based), `step`, `alter`, `octave`, `ks_fifths`.
- **Notational identity:** `staff`, `voice`, `is_note_onset` (False on tied continuations), `mc`, `mn`, `section_start`.
- **Harmonic labels — the `a_*` family, engineered to be cross-corpus comparable:** `a_simpleNumeral`, `a_romanNumeral`, `a_degree1`, `a_degree2`, `a_quality`, `a_inversion`, `a_root`, `a_bass`, `a_localKey`, `a_tonicizedKey`, `a_phraseend`, `a_isOnset`. Plus contextual: `note_degree`, `cadence_type`, `pedal`, `tpc_is_in_label`, `tpc_is_root`, `tpc_is_bass`.
- **Validity masks** (for selective filtering): `valid_chord_label`, `valid_cadence_label`, `valid_phrase_label`, `valid_pedal_point_label`, `valid_section_start_label`.

The authoritative column dictionary lives in `processing/DLC/dlc_pitch_array_specs.csv` and `processing/DLC/dlc_specs_specs.json`. Consider rendering a slice of that as a table on the poster (e.g., 10 most informative columns).

### 5.3 The schema excerpt — *featured panel*

The single most poster-friendly artifact in the codebase is `dlc_pitch_array_specs.csv` (included in the upload bundle). Its `used_for` column tells you *at a glance* which downstream task each field trains. Render this as a poster panel — it carries the multi-task story without any prose, and it's the bridge between the musicology audience and the ML audience.

**Suggested visual treatment.** A four-column table (`column` · `dtype` · `used_for` · `description`), grouped by `used_for` value with subtle background-tint bands, ordered to walk the reader from low-level note features → metrical labels → harmonic labels → filters. Color-code the `used_for` categories so the eye can scan by task.

A curated 21-row excerpt that tells the whole story (out of ~90 columns in the full schema):

| column | dtype | used_for | description (abbreviated) |
| --- | --- | --- | --- |
| `onset_div` | Int64 | input graph creation | Proportional integer position |
| `duration_div` | Int64 | input graph creation | Proportional integer duration |
| `pitch` | Int64 | input graph creation | MIDI value |
| `tpc` | Int64 | computing fields | Tonal Pitch Class (0=C, −1=F, 1=G, …) |
| `step` | string | input graph creation | Note name without accidental (A–G) |
| `alter` | Int64 | input graph creation | Note accidental: [−3, 3] |
| `ks_fifths` | Int64 | input graph creation | Key signature: [−7, 7] |
| `is_note_onset` | boolean | input graph creation | False when tied to a previous note |
| `downbeat` | Int64 | **training beat inference** | Integer beat positions, 0 elsewhere |
| `is_downbeat` | boolean | **training beat inference** | True on downbeats |
| `cadence_type` | string | **training cadence induction** | Cadence label ∈ {PAC, IAC, HC, EC, DC, PC} |
| `a_phraseend` | boolean | **training phrase inference** | True at structural phrase ends |
| `section_start` | boolean | **training section inference** | True at section / repeat boundaries |
| `note_degree` | string | **training note-degree inference** | Note's position in the local-key scale |
| `a_degree1` | string | **training harmony inference** | Chordal root as scale degree |
| `a_degree2` | string | **training harmony inference** | Tonicized key as scale degree (applied chords) |
| `a_quality` | string | **training harmony inference** | Chord quality (music21 vocabulary) |
| `tpc_is_root` | boolean | **training harmony inference** | Note pitch-class == chord root |
| `tpc_is_bass` | boolean | **training harmony inference** | Note pitch-class == chord bass |
| `valid_chord_label` | boolean | **filtering** | A valid chord label is available |
| `valid_cadence_label` | boolean | **filtering** | A cadence label is available |

The nine distinct `used_for` values map directly onto AnalysisGNN's task heads:

- *input graph creation* → node features for the GNN
- *input graph metadata and informational purposes* → identification (`mc`, `mn`, `octave`, …)
- *computing fields* → intermediate values, dropped before training
- *training **beat** inference* → metrical analysis
- *training **cadence** induction* → cadence detection (`PAC`, `HC`, …)
- *training **harmony** inference* → Roman numeral analysis
- *training **phrase** inference* → phrase segmentation
- *training **section** inference* → structural segmentation
- *training **note_degree** inference* → scale-degree-relative-to-key
- *filtering* → validity masks so models train only on annotated regions

This panel is where the corpus stops being abstract and starts being *concretely useful*. It also implicitly answers "what would I do with this?" without requiring the viewer to read any prose.

**Optional callout — the `a_simpleNumeral` regex (Listing 1 in the DLfM paper).** Tighter and more cryptic; include only if there's space for a target-vocabulary box.

```text
^(
  (?P<acc>#+|b+)?
  (?P<root>Cad|Ger|It|Fr|N|
           VII|VI|V|IV|III|II|I|
           vii|vi|v|iv|iii|ii|i)
)
(?P<quality>[+o%])?
(?:maj|#|b|M)?
(?P<seven>7)?
(?:[#bM])?
(?P<nine>9)?$
```

`a_simpleNumeral` is the *minimal* harmonic label: chordal root + sonority only. Inversion, extensions, and applied/secondary degree live in separate columns (`a_inversion`, `a_degree2`, etc.) so the user can opt in.

### 5.4 The Cadential 6/4 dilemma (great single-slide story)

The poster's best single illustration of "why this is hard":

| Standard | A cadential six-four is annotated as | Implicit interpretation |
| --- | --- | --- |
| RomanText / AugmentedNet | `I64` or `Cad64` | Tonic chord, second inversion |
| DCML / DLC | `V(64)` | Dominant with double suspension (4–3, 6–5) |

Same notes. Same musical function. *Categorically different chord roots.* dilemmadata rewrites both to a shared symbol **`Cad`**. It is the **seventh most frequent harmony** in the joined output — not a corner case.

(Worth flagging visually: this is the kind of disagreement that silently corrupts cross-corpus model training.)

### 5.5 Other reconciliation specifics (smaller panel / footnote material)

- **Chord-type vocabulary** mapped to music21 conventions. E.g., DCML `Mm7` → `dominant seventh chord`, `%7` → `half-diminished seventh chord`, `Ger` → `German augmented sixth chord`. See `DLC_CHORD_TYPE_MAPPING` in `processing/utils.py`.
- **Inversion vocabulary** mapped to integers 0–3 via `FIGBASS2INVERSION` (`6`→1, `65`→1, `64`→2, `43`→2, `42`→3, `2`→3).
- **Minor-mode VI/VII** — DCML defaults to natural minor (so `VII` and `viio` share a root); RomanText infers from chord type. `add_simpleNumeral_columns` handles the cross-walk.
- **Bypassing music21.chordify** on the AN side: upstream AugmentedNet emits one row per fixed 16th-note "Salami slice" (~100K rows). The fork emits one row per *note event* (~750K rows), enabling true note-wise (rather than slice-wise) learning. **This is a structural change, not a cosmetic one** — worth calling out on the poster as a methodological contribution.
- **Validity masks** for labels that can't be reliably converted (e.g., rare extensions, inversion notation at phrase boundaries) — included as columns so downstream users can filter.
- **Provenance hashes**: each piece carries the Git commit hash of the file it was last derived from, plus a constructed URL pointing at the exact upstream version. Lets you re-resolve to source ground truth.

### 5.6 The pipeline (this is the headline figure)

Left-to-right flow with **five vertical bands of color**:

1. **Source meta-corpora** (gray boxes): a stack of small labels (ABC v1.0/v2.5, BPS, TAVERN, WiR, WTC, HaydnSun, …) feeding into the **AugmentedNet v1.9.1** box and the **Distant Listening Corpus v3.1** box. A connector showing the 99-piece overlap (highlighted: 70 from ABC).
2. **Music representations** (green): score files + annotation files for AN; annotated MuseScore files for DLC.
3. **Parsing libraries** (orange): `music21` for AN, `ms3` + `dimcat` for DLC. Python logo on both.
4. **Transformation / homogenization** (red): bullet list — adding features, transforming/reducing features, converting representations.
5. **Output** (blue): **dilemmadata v1.1**, a single shared CSV/TSV representation with the note-wise schema.

The upload bundle includes `dilemmadata.drawio` (editable source), `dilemmadata.drawio.pdf` (rendered), and `dilemmadata.svg` (a cleaner SVG rendering already prepared). **For the poster, redraw cleanly at high DPI** — the existing diagram is functional but visually utilitarian. Consider a more poster-friendly horizontal layout with strong typography.

### 5.7 The Beethoven worked example (great visual centerpiece)

Beethoven, *Piano Sonata No. 3* in C major, Op. 2 No. 3, first movement, **mm. 10–13**. This piece is one of the 84 overlap pieces with distinct analyses. The figure stacks four staves:

1. **Score** (two staves, piano grand staff).
2. **AugmentedNet analysis** layer below the score (lower granularity, uses `I64` for the cadential six-four).
3. **Distant Listening Corpus analysis** layer (higher granularity, uses `V(64)` with suspension notation).
4. **dilemmadata simplified** layer (both analyses mapped to the shared `a_simpleNumeral` vocabulary, including the special `Cad` symbol; congruent symbols printed in black, divergent in color).

This is the single most pedagogically effective visual in either paper. It shows the *cost* of the dilemma (the two source rows visibly disagree), the *resolution* (the bottom row unifies them), and the *residual disagreement* (where black turns to color).

Source files in the upload bundle: `beethoven_annotation_comparison_excerpt.png` and `beethoven_annotation_comparison_excerpt.svg` (rendered PNG / SVG of the four-stave figure).

### 5.8 The 84-piece "dilemma" panel

Suggested standalone panel — small but conceptually important. A visual showing the Venn-diagram-ish relationship:

- AND (353) — DLC (1,268) — overlap (99) — kept overlap (84).
- Annotated under *both* RomanText and DCML.
- Proposed in the papers as a benchmark for *paradigm disagreement* studies, and as a "catalyst for replacing the prevailing reliance on objective 'ground truth' in training scenarios."

Quote box candidate (from the MEC paper conclusion):

> "We propose the overlapping dilemma pieces, annotated under both source paradigms, as a catalyst for replacing the prevailing reliance on objective 'ground truth' in training scenarios."

### 5.9 Future work (right-edge panel, compact)

- **`flexohr`** — planned library that generalizes this two-corpus prototype into a flexible loader for *any* heterogeneous Roman-numeral corpus, producing user-selected target representations (tabular, root-only, full intervallic, key-resolved, etc.).
- **Catch up to current When-in-Rome.** Most AND content originates in MarkGotham's *When-in-Rome* corpus, which has continued to grow; pulling in newer WiR pieces is a future-work item.
- **Context-sensitive validity metrics** — grounded in music theory, quantifying how appropriate an annotation is given its surrounding context. Enables multi-label prediction and systematic comparison of competing analyses.

### 5.10 Acknowledgments / citation block

The poster needs a footer with:
- Project URL (whenever public).
- Zenodo DOI (a `.zenodo.json` exists in the repo — there's a Zenodo deposition).
- AnalysisGNN as the downstream consumer that motivated the schema design — cite Karystinaios et al. (CMMR 2025) and the GitHub link (https://github.com/manoskary/analysisgnn).
- Source corpus citations: Nápoles López et al. (ISMIR 2021) for AugmentedNet; Hentschel, Rammos, Neuwirth, Rohrmeier (Scientific Data 2025) for DLC.

### 5.11 Reproducibility & the shared test set (optional panel)

`Dilemmadata_merged_summary.pdf` (in the upload bundle) is a corpus-wide map: **every piece** in the merged dataset shown as one row, with **all DLC pieces on the left**, **all AN pieces on the right**, and **concurrent (overlapping) pieces aligned on the same row**. A middle column shows which training labels each piece carries (chord / cadence / phrase / pedal …), and a dedicated column highlights **test-set pieces in red** — these are pieces that may *never* be used for training, exposed publicly so that any future model trained on dilemmadata can be compared apples-to-apples against others.

Why this matters on the poster:

- It's the only single image that shows the *whole* corpus at a glance — useful as a "data at scale" visual.
- It makes the **84 overlapping pieces** visible as a vertical band of side-by-side rows.
- It surfaces the **frozen public test set** — an explicit invitation to the community to use dilemmadata as a benchmark, not just a training resource. Comparability across future papers depends on this set being respected.

**If space allows**, include this either as a half-column thumbnail with a "scan the QR for the full table" treatment, or — if the poster is large enough — as a full-bleed band along the bottom of the poster, where the alignment story reads at a glance.

---

## 6. Stuff I deliberately omitted from CLAUDE.md but want the poster to know

Backstory + motivation that doesn't help future coding agents but *does* help a design audience grasp the stakes:

- **The motivation is downstream model training, not curation for its own sake.** dilemmadata was originally produced as training data for AnalysisGNN — a graph neural network for multi-task music analysis (cadence detection, phrase segmentation, key analysis, harmonic analysis, voice leading, section segmentation, pedal-point detection, note-degree inference). The schema was reverse-engineered from those task heads as much as from the source corpora.
- **The "interoperability isn't just syntactic" argument.** From the MEC paper: discrepancies between Roman-numeral datasets produced under disparate paradigms "extend far beyond mere syntax; they create severe impediments to integration and interoperability. The substantial effort required to carefully transform annotations for equivalence without distorting the original semantic intent underscores […] the urgent need for a generalised data model capable of interfacing with the diverse harmony encoding standards currently in use."
- **There is no objective ground truth.** Harmony analysis is interpretive; no metric exists for how well a label fits its musical context across competing theoretical assumptions. The papers explicitly frame this as a *call to the community*. The dilemmadata overlap subset is the proposed experimental substrate.
- **AugmentedNet itself was already heterogeneous.** It aggregates manual and automated annotations from six different upstream corpora. So even before dilemmadata, "AugmentedNet ground truth" was already a smoothed-over consensus, not a single authoritative voice. This blunts the apparent asymmetry of "DLC trained vs. AND annotated."
- **The music21 version dilemma.** The AN v1.0.0 *dataset* was authored against music21 v6.7.0. The current `dimcat` dependency pulls music21 v9.5.0, which is *stricter* — e.g., it no longer accepts "beat 3" in a 3/8 meter without the explicit "slow 3/8" syntax. Rather than pull in newer When-in-Rome materials wholesale (the principled long-term plan), the AN fork (now at v1.9.1) applies the *minimum-necessary* syntax patches to v1.0.0 data so it still parses under modern music21. A self-aware band-aid.
- **Stripping the TensorFlow dependency.** The dilemmadata branch of AugmentedNet drops AugmentedNet's TensorFlow dependency, which is heavy and unused for the dataset-production path.
- **The repo doubles as a methodology paper artifact.** Two manuscripts (DLfM '25 long-form, MEC 2026 short-form) document the design decisions and serve as the authoritative spec for *why* particular columns and transformations exist.
- **"Dilemmadata" as wordplay.** The name encodes that producing the data was itself a sequence of dilemmas — and that the overlap subset *contains* dilemmas as first-class objects rather than hiding them behind majority votes.

---

## 7. Suggested poster layout (rough zones)

A standard A0 portrait or 36"×48" landscape would accommodate this. Sketch:

```
+-----------------------------------------------------------+
|              DILEMMADATA — title + tagline                |
| authors / affiliations / DOI strip                        |
+-------------------+----------------------+----------------+
| 1. THE PROBLEM    | 2. THE PIPELINE      | 3. THE         |
|  - two corpora    |    (big horizontal   |    DILEMMA     |
|  - three axes of  |     flow diagram)    |  - 84 overlap  |
|    incompatibility|                      |    pieces      |
|  - Cad 6/4 table  +----------------------+  - quote box   |
|                   | 4. WORKED EXAMPLE    |  - paradigm    |
| Headline stats    |    (Beethoven        |    comparison  |
| (1,621 / 2.8 M)   |     Op. 2 No. 3,     +----------------+
|                   |     mm. 10-13)       | 5. FUTURE WORK |
|                   |                      |  - flexohr     |
|                   |                      |  - WiR catchup |
+-------------------+----------------------+----------------+
| 6. SCHEMA SNAPSHOT (column dictionary)   |  7. REGEX BOX  |
|    + chord-type / inversion mappings     |    + chord-fn  |
+------------------------------------------+----------------+
| Citation strip · Zenodo DOI · QR to repo · acknowledgments |
+-----------------------------------------------------------+
```

DOI: https://doi.org/10.5281/zenodo.19661223

## 8. Design notes & aesthetic guidance

- **Color palette.** The existing pipeline diagram uses a five-tone scheme: gray (sources), green (musical representations), orange (parsers), red (transformations), blue (output). This is a *good* mnemonic — preserve it. Saturate the colors more for poster legibility; consider a deep teal for the output band instead of pure blue.
- **Typography.** Music notation needs serif or musicological-feeling type (e.g., a chord-symbol-friendly serif). Body text can be a clean sans (Inter, IBM Plex Sans). Roman numerals should be *italicized* with the same conventions as the source corpora (uppercase for major, lowercase for minor). The `Cad`, `Ger`, `It`, `Fr`, `N` literals should be set in a slightly different style (small-caps or condensed) to mark them as special symbols.
- **Mathematical / regex callouts.** Use a monospace box with a faint background tint for the Listing-1 regex and any chord-mapping tables.
- **Music notation reproductions.** The Beethoven excerpt is the visual anchor; render it large enough that someone two meters away can read the chord symbols beneath the staves.
- **Avoid over-decoration.** This is a research poster, not a marketing piece. Whitespace and clear hierarchy beat gradients.
- **Logos.** Whichever institutional logos are appropriate (likely Bruckneruniversität Linz given the user's email; possibly EPFL DCML; possibly McGill via AugmentedNet lineage — confirm with author).
- **One thing to *not* do:** don't try to render the full 2.8M-row TSV as a heatmap or similar. The data is too sparse along most axes to make a coherent overview viz. Stick to the worked Beethoven example, which is rhetorical gold.

## 9. Visual material inventory (the `design/` upload bundle)

All files referenced below live in `design/` next to this brief. Paths are relative to that folder.

| File | What it shows | Suitable for |
| --- | --- | --- |
| `MEC2026_Dilemmadata.pdf` | The short paper (13 pages incl. references) | Reference for designer; primary source for narrative + figures |
| `manuscript.tex` | Long-form paper source (DLfM '25) — more detail than MEC; contains Listing 1 regex inline | Background reading; authoritative for design rationale |
| `dilemmadata.bib` | Bibliography source | Citation strip |
| `beethoven_annotation_comparison_excerpt.png` | Beethoven Op. 2 No. 3 mm. 10–13 with AN / DLC / dilemmadata-simplified analyses stacked | **Centerpiece worked example** |
| `beethoven_annotation_comparison_excerpt.svg` | Vector version of the same | Print quality |
| `dilemmadata.drawio` | Editable source of the pipeline diagram (open in diagrams.net) | Re-style and re-render the pipeline figure |
| `dilemmadata.drawio.pdf` | PDF rendering of the same | Drop-in figure if no redraw needed |
| `dilemmadata.svg` | Cleaner SVG rendering of the pipeline | Print quality |
| `dlc_pitch_array_specs.csv` | Authoritative schema column dictionary with dtypes + descriptions + `used_for` task tags | **Source for the schema-snapshot panel (§5.3)** |
| `Dilemmadata_merged_summary.pdf` | Whole-corpus alignment table: all DLC pieces (left) / all AN pieces (right), concurrent pieces on shared rows, label-availability middle column, test-set pieces highlighted in red | **Source for the reproducibility / test-set panel (§5.11)** — also a strong "data at scale" hero visual |
| `03-1_rntext_analysis.txt` | The `.rntxt` analysis file for the Beethoven excerpt | "What an .rntxt file looks like" callout box |
| `doi_qrcode.png` | QR code linking to the Zenodo deposition | Citation strip |

### Not in the upload bundle but worth knowing about

Generated by running the pipeline, only relevant if the designer wants a real data sample for layout reference:

- `pitch_arrays/AN/{training,test,validation}/*.tsv` — 353 AN pieces in shared schema.
- `pitch_arrays/DLC/<subcorpus>/*.tsv` — 1,268 DLC pieces in shared schema.
- `pitch_arrays/DLC/beethoven_piano_sonatas/01-1.tsv` — small enough to be legible at a glance; share on request.

---

## 10. Files to upload to Claude Design

**Minimum viable upload bundle (everything Claude Design needs to compose the poster):**

0. **`reference_poster.pptx`** — the reference poster serving as Design System in terms of logos, fonts, visual language, etc.
1. **`design_prompt.md`** — this file. The brief.
2. **`MEC2026_Dilemmadata.pdf`** — the short paper. Self-contained narrative + numbers + figures embedded.
3. **`beethoven_annotation_comparison_excerpt.png`** (or `.svg`) — the worked-example centerpiece.
4. **`Dilemmadata_merged_summary.pdf`** — whole-corpus alignment table (see §5.11). Hero "data at scale" visual; also shows the public test set.
5. **`dilemmadata.drawio`** — pipeline diagram source (can be opened in diagrams.net / drawio for re-styling). If the designer can't open `.drawio`, export it to PDF/PNG first.
6. **`README.md`** — for top-level project framing.
7. **`processing/DLC/dlc_pitch_array_specs.csv`** — schema column dictionary, so the designer can render a clean table of the schema.

**Strongly recommended additions (richer, after `git submodule update --init --recursive`):**

8. **`dlfm_paper/manuscript.tex`** — long-form paper. More detail than MEC; includes Listing 1 regex inline.
9. **`dlfm_paper/figures/dilemmadata.drawio.pdf`** — clean PDF render of pipeline.
10. **`doi_qrcode.png`** — large QR code linking to the Zenodo deposition, for the citation strip.
11. **`dlfm_paper/figures/cad64.pdf`** — secondary illustration of the I64 / V(64) ambiguity.
12. **`dlfm_paper/figures/03-1_rntext_analysis.txt`** — a real RomanText file, good for a "what an .rntxt file looks like" callout.
13. **`dlfm_paper/dilemmadata.bib`** — bibliography source for the citation strip.

**Optional context (only if the designer asks for more depth):**

14. **`CLAUDE.md`** — technical orientation; explains what the codebase looks like.
15. **`processing/utils.py`** — source of the chord-type mappings, the cadential V rewrite, the regex constants.
16. **`pitch_arrays/DLC/beethoven_piano_sonatas/01-1.tsv`** — concrete sample of the output schema in real data, for layout reference.
17. **`.zenodo.json`** — for the exact authors / DOI / keywords in the citation strip.

**Do NOT upload:**

- The corpora submodules (`corpora/AugmentedNet/`, `corpora/distant_listening_corpus/`). Multi-GB and irrelevant for design.
- The generated `pitch_arrays/` TSVs (other than a single sample on request).
- The MuseScore corpora binaries.

---

## 11. Quick-reference: things the designer might get wrong

- **It's "Roman numeral analysis," not "roman numeral analysis"** — proper noun for the harmonic-analytical tradition.
- **The shared symbol for the cadential six-four is `Cad`, not `Cad64`** in the dilemmadata vocabulary. `Cad64` is a RomanText artifact only.
- **AND ≠ "and"** in this context. It stands for "AugmentedNet Dataset" and is consistently italicized or set in small caps in the papers.
- **AnalysisGNN is one word, mixed case.**
- **"Distant Listening Corpus" is the full name; DLC is the abbreviation.** Not "DLCorpus" or "Distant-Listening".
- **`a_*` prefix** on columns is functional, not decorative — the `a` stands for "annotation" / "aligned" and the prefix marks columns engineered to be comparable across both source corpora. Worth a footnote in the schema panel.
- **The `.rntxt` file extension is lowercase**, and music21 spells it "RomanText" (one word, mixed case) when referring to the standard.
- **MuseScore is one word, mixed case.**

---

*End of brief. If anything here is ambiguous or under-specified, ask before composing — better to clarify than to ship a poster with a wrong claim about the data.*
