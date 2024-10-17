# CLaMP 2 Codebase

## Overview
CLaMP 2 is a state-of-the-art multimodal music information retrieval system designed to work with 101 languages. This codebase includes scripts for training models, extracting features, and utility functions for processing music and text data. Below is a description of the scripts contained in the `code/` folder.

## Repository Structure
The `code/` folder contains the following scripts:

### 1. `config.py`
This script contains the training hyperparameters and file paths used in the `train_clamp2.py` and `train_m3.py` scripts. You can modify parameters such as learning rates, batch sizes, and file locations for training data.

### 2. `extract_clamp2.py`
This script utilizes the pre-trained CLaMP 2 model to extract representations of text (.txt) or music (.abc or .mtf) from a specified input folder and save the features to a target output folder in `.npy` format. The extracted features can be normalized for semantic search or retain temporal information for classification tasks.

**Usage:**
```bash
python extract_clamp2.py <input_dir> <output_dir> [--normalize]
```
- `input_dir`: Directory containing input data files.
- `output_dir`: Directory to save the output features.
- `--normalize`: (Optional) Normalize the extracted features. Normalization is not required for music classification tasks, but it is required for semantic search tasks.

### 3. `extract_m3.py`
This script employs the pre-trained M3 model to extract representations in interleaved ABC notation and MIDI Text Format (MTF) from the specified input folder, saving the features to the target folder as `.npy` files.

**Usage:**
```bash
python extract_m3.py <input_dir> <output_dir>
```
- `input_dir`: Directory with input files (in .abc or .mtf format).
- `output_dir`: Directory to save extracted features.

### 4. `train_clamp2.py`
This script manages the training process for the CLaMP 2 model. It prepares training data from a path specified in the `TRAIN_JSONL` variable, which is defined in the `config.py` file. If `EVAL_JSONL` is provided in the configuration, it will be used for validation. By default, 1% of the training data is reserved for validation.

CLaMP 2 utilizes the multilingual text encoder `FacebookAI/xlm-roberta-base` for processing text data. Additionally, it employs the M3 model, pre-trained on both ABC and MIDI data, as the multimodal music encoder. If the pre-trained weights for M3 are available and the configuration variable `CLAMP2_LOAD_M3` is set to True, the training script will automatically load the M3 weights.

**Training Command:**
To start the training process, use the following command:

```bash
torch.distributed.launch --nproc_per_node=<number_of_GPUs> --use_env train_clamp2.py
```

Replace `<number_of_GPUs>` with the number of GPUs you want to use for training.

**Input Data Format**
The input training data should be in JSONL format, where each line contains a single JSON object with the following structure. Fields that do not apply should be set to `None`:

```json
{
  "title": "Song Title",
  "composer": "Composer Name",
  "genres": ["Genre1", "Genre2"],
  "description": "Song description.",
  "lyrics": "Song lyrics.",
  "tags": ["tag1", "tag2"],
  "ensembles": ["Ensemble Name"],
  "instruments": ["Instrument1", "Instrument2"],
  "summary_en": "English summary.",
  "summary_nen": {
    "language": "Language Name",
    "summary": "Summary in specified language."
  },
  "filepaths": [
    "path/to/abc/file.abc",
    "path/to/mtf/file.mtf"
  ]
}
```

For obtaining the English and non-English summaries generated by GPT-4, refer to the `process_data/gpt4_summarize.py` script.

### 5. `train_m3.py`
This script is dedicated to training the M3 model using interleaved ABC and MTF files. The directories for training and optional evaluation data should be specified in the `TRAIN_FOLDERS` and `EVAL_FOLDERS` variables, respectively.

**Training Command:**
To start the training process for the M3 model, use the following command:

```bash
torch.distributed.launch --nproc_per_node=<number_of_GPUs> --use_env train_m3.py
```

Replace `<number_of_GPUs>` with the number of GPUs you want to use for training.

**Data Preparation:**  
The data should be structured in interleaved ABC (.abc) and MTF (.mtf) formats. Please refer to the `process_data/` folder for instructions on how to prepare these formats.

### 6. `utils.py`
This utility script contains various classes for model definitions and functions used for training.