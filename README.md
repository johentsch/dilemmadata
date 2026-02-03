# dilemmadata

**A Unified Dataset for Multi-Task Music Analysis with Graph Neural Networks**

This repository contains the processed and aligned data from two major annotated music corpora: the **AugmentedNet** dataset and the **Distant Listening Corpus (DLC)**. The data has been preprocessed and converted into **pitch arrays** — tabular representations suitable for graph-based machine learning models used in automated music analysis tasks.

---

## Overview

This project serves as the data infrastructure for training graph neural networks (GNNs) on multiple music analysis tasks, including:

- **Cadence detection** (identifying cadence types in musical passages)
- **Phrase segmentation** (marking phrase boundaries)
- **Key analysis** (local and global key detection)
- **Harmonic analysis** (chord quality, inversion, root, bass note)
- **Roman numeral analysis** (functional harmonic analysis)
- **Rhythmic analysis** (downbeat and metrical analysis)
- **Voice leading** (analysis of voice leading patterns)
- **Section segmentation** (identifying structural sections)
- **Pedal point detection** (sustained bass notes)
- **Note degree inference** (scale degrees relative to local key)

This resource has been demonstrated through the **AnalysisGNN** framework ([github.com/manoskary/analysisgnn](https://github.com/manoskary/analysisgnn)) and serves as a foundation for training neural networks on automated music analysis tasks using multi-task learning and graph-based representations.

---

## Data Sources

### 1. AugmentedNet Dataset
**Source:** [github.com/napulen/AugmentedNet](https://github.com/napulen/AugmentedNet)

AugmentedNet is an automatic Roman numeral analysis neural network developed by Néstor Nápoles López as part of his PhD research. The dataset includes:

- **353 pieces** from multiple collections (Beethoven Piano Sonatas, Bach chorales, TAVERN, etc.)
- Roman numeral annotations for harmonic analysis
- MusicXML scores with RomanText annotations
- **Split**: Pre-defined test/training/validation splits (v1.0.0 dataset)

**Key features:**
- Cadence annotations (cadential labels)
- Roman numeral analysis (functional harmony)
- Chord annotations with inversions
- Synthetic training examples via texturization

**Reference:**
> Nápoles López, N., Gotham, M., & Fujinaga, I. (2021). AugmentedNet: A Roman Numeral Analysis Network with Synthetic Training Examples and Additional Tonal Tasks. In *Proceedings of the 22nd International Society for Music Information Retrieval Conference* (pp. 404–411). https://doi.org/10.5281/zenodo.5624533

### 2. Distant Listening Corpus (DLC)
**Source:** [github.com/DCMLab/distant_listening_corpus](https://github.com/DCMLab/distant_listening_corpus)

The Distant Listening Corpus is a large-scale collection of annotated musical scores from the DCML (Digital and Cognitive Musicology Lab) corpus initiative. It includes over **40 subcorpora** spanning music from the 17th to 20th centuries:

- Bach, Beethoven, Chopin, Mozart, Schubert, etc.
- Comprehensive harmonic annotations using the DCML standard
- MuseScore 3.6.2 files with embedded annotations
- TSV exports of notes, measures, chords, and harmony labels

**Included subcorpora** (selected):
- `beethoven_piano_sonatas`, `chopin_mazurkas`, `mozart_piano_sonatas`
- `bach_en_fr_suites`, `bach_solo`, `schubert_winterreise`
- `debussy_suite_bergamasque`, `grieg_lyric_pieces`, `liszt_pelerinage`
- `monteverdi_madrigals`, `scarlatti_sonatas`, `wagner_overtures`
- And many more...

**Key features:**
- Phrase boundaries
- Cadence annotations
- Local and global key annotations
- Pedal point annotations
- Section start markers
- Note degree annotations (scale degree relative to local key)

**Reference:**
> Hentschel, J., Rammos, Y., Neuwirth, M., & Rohrmeier, M. (2025). A corpus and a modular infrastructure for the empirical study of (an)notated music. *Scientific Data*, 12(1), 685. https://doi.org/10.1038/s41597-025-04976-z

---

## Repository Structure

```
dilemmadata/
├── README.md                          # This file
├── corpora/                           # Original corpus data (git submodules)
│   ├── AugmentedNet/                  # AugmentedNet raw data
│   └── distant_listening_corpus/      # DLC raw data
├── pitch_arrays/                      # Processed pitch array representations
│   ├── AN/                            # AugmentedNet pitch arrays
│   │   ├── test/                      # Test split
│   │   ├── training/                  # Training split
│   │   ├── validation/                # Validation split
│   │   └── dataset_summary.tsv        # Metadata summary
│   └── DLC/                           # Distant Listening Corpus pitch arrays
│       ├── beethoven_piano_sonatas/   # Organized by subcorpus
│       ├── chopin_mazurkas/
│       ├── mozart_piano_sonatas/
│       └── ...                        # 40+ subcorpora
└── processing/                        # Processing scripts and utilities
    ├── utils.py                       # Core utility functions
    ├── requirements.txt               # Python dependencies
    ├── merged_summary.tsv             # Merged metadata from both corpora
    ├── augnet_summary_v100.tsv        # AugmentedNet v1.0.0 metadata
    ├── dlc_summary.tsv                # DLC metadata
    ├── AN/                            # AugmentedNet processing scripts
    │   ├── create_pitch_arrays.py     # Generate pitch arrays from AN
    │   ├── concat_pitch_arrays.py     # Concatenate all AN arrays
    │   ├── data_overview.py           # Compile AN metadata
    │   ├── test_transformation_equivalence.py  # Validation checks
    │   └── ...
    ├── AN_mscx/                       # Assembled MuseScore files from AN
    │   └── labels/
    └── DLC/                           # DLC processing scripts
        ├── create_pitch_arrays.py     # Generate pitch arrays from DLC
        ├── design_test_split.py       # Design test split alignment
        ├── dlc_pitch_array_specs.csv  # Column specifications
        └── ...
```

---

## Pitch Array Format

A **pitch array** is a tabular representation of a musical score where each row represents a note, and columns contain features relevant to music analysis tasks. This format bridges symbolic music representations and graph-based machine learning models.

### Core Columns (Input Features)

| Column | Description | Data Type |
|--------|-------------|-----------|
| `onset_div` | Proportional integer position (in divisions) | Int64 |
| `duration_div` | Duration in divisions | Int64 |
| `onset_beat` | Beat position as a fraction | Fraction/Float |
| `pitch` | MIDI pitch number | Int64 |
| `tpc` | Tonal Pitch Class (fifths: C=0, G=1, F=-1) | Int64 |
| `step` | Note step (C, D, E, F, G, A, B) | String |
| `alter` | Chromatic alteration (#=1, b=-1) | Int64 |
| `beat_float` | Floating-point beat position | Float64 |
| `downbeat` | Downbeat position in measure | Int64 |
| `is_downbeat` | Boolean flag for downbeat | Boolean |
| `ts_beats` | Time signature numerator | Int64 |
| `ts_beat_type` | Time signature denominator | Int64 |
| `staff` | Staff number | Int64 |
| `voice` | Voice number | Int64 |

### Label Columns (Task Targets)

**From AugmentedNet:**
- `a_simpleNumeral`, `a_romanNumeral`: Roman numeral analysis
- `a_degree1`: Chord degree
- `a_inversion`: Chord inversion
- Cadence information

**From Distant Listening Corpus:**
- `chord`: Chord label (DCML standard)
- `cadence`: Cadence type
- `phrase`: Phrase annotation
- `localkey`, `globalkey`: Key signatures (as tonal pitch classes)
- `localkey_is_minor`, `globalkey_is_minor`: Key mode
- `pedal`: Pedal point annotation
- `section_start`: Section boundary marker
- `note_degree`: Scale degree relative to local key

### Purpose

The pitch array format enables:
1. **Graph construction**: Notes become nodes; temporal, harmonic, and hierarchical relationships become edges
2. **Multi-task learning**: Different columns serve as targets for different analysis tasks
3. **Data alignment**: A common format for diverse corpora with different annotation standards
4. **Efficient processing**: TSV format for fast loading and manipulation

---

## Data Alignment Process

The two corpora were carefully aligned to create a unified training dataset while maintaining a clean test set:

### 1. Metadata Compilation
**Script:** `processing/AN/data_overview.py`

- Extract metadata from both corpora
- Identify overlapping pieces
- Generate summary files:
  - `augnet_summary_v100.tsv` — AugmentedNet metadata
  - `dlc_summary.tsv` — DLC metadata
  - `merged_summary.tsv` — Combined metadata with overlap information

### 2. Test Split Design
**Script:** `processing/DLC/design_test_split.py`

- **Exclusion rule**: Any DLC piece that overlaps with the AugmentedNet v1.0.0 test set is excluded from training
- This ensures no data leakage between test and training sets
- Result: 2 pieces excluded from DLC (Chopin Mazurkas BI61-5op07-5, BI77-3op17-3)

### 3. Validation
**Script:** `processing/AN/test_transformation_equivalence.py`

- Verify that pitch array transformations are consistent
- Check data integrity across splits
- Validate column specifications and data types

### 4. Pitch Array Generation

**AugmentedNet:**
```bash
# Generate pitch arrays with train/test/validation splits
python processing/AN/create_pitch_arrays.py
```

**Distant Listening Corpus:**
```bash
# Generate pitch arrays for all subcorpora
python processing/DLC/create_pitch_arrays.py
```

---

## Column Specifications

Each pitch array comes with a **specification file** (CSV or JSON) that describes:
- Column names
- Data types
- Purpose (input feature, training label, metadata, etc.)
- Description of each field

**Example:**
- `processing/DLC/dlc_pitch_array_specs.csv` — DLC column specifications
- `processing/DLC/dlc_specs_specs.json` — Metadata about the specifications

These files follow the [Frictionless Data](https://specs.frictionlessdata.io/) standard and can be used for validation and automated loading.

---

## Related Projects

### AnalysisGNN
**Repository:** [github.com/manoskary/analysisgnn](https://github.com/manoskary/analysisgnn)

A comprehensive framework for multi-task music analysis using Graph Neural Networks. Supports:
- HybridGNN, HGT, MetricalGNN architectures
- Continual learning for sequential task acquisition
- Pre-trained models available via Weights & Biases

**Reference:**
> Karystinaios, E., Hentschel, J., Neuwirth, M., & Widmer, G. (2025). AnalysisGNN: A Unified Music Analysis Model with Graph Neural Networks. In *International Symposium on Computer Music Multidisciplinary Research (CMMR)*.

### AugmentedNet
**Repository:** [github.com/napulen/AugmentedNet](https://github.com/napulen/AugmentedNet)

Neural network for automatic Roman numeral analysis with synthetic data augmentation.

### Distant Listening Corpus
**Repository:** [github.com/DCMLab/distant_listening_corpus](https://github.com/DCMLab/distant_listening_corpus)

A modular infrastructure for the empirical study of annotated music, maintained by the Digital and Cognitive Musicology Lab (DCML).

---

## Acknowledgments

This work builds upon:
- The **AugmentedNet** dataset by Néstor Nápoles López
- The **Distant Listening Corpus** by the DCML Lab

