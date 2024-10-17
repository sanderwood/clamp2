# Music Classification Codebase

## Overview
Linear Probe is a powerful classification tool that leverages feature representations for supervised learning tasks. This codebase includes scripts for training a linear classification model, performing classification on new feature data. The features utilized can be extracted from the M3 or CLaMP 2 models, ensuring that the time dimension information is preserved and **not normalized**. Below is a description of the scripts contained in the `music_classification/` folder.

## Repository Structure
The `music_classification/` folder contains the following scripts:

### 1. `config.py`
This script defines configurations for the linear probe training and inference, specifying training data paths and parameters like learning rate, number of epochs, and hidden size.

### 2. `inference_cls.py`
This script enables the classification of feature vectors using a pre-trained linear probe model.

#### JSON Output Format
The resulting JSON file contains a dictionary with the following structure:
```json
{
    "path/to/feature1.npy": "class_A",
    "path/to/feature2.npy": "class_B",
    "path/to/feature3.npy": "class_A"
}
```
- **Key**: The path to the input feature file (e.g., `feature1.npy`).
- **Value**: The predicted class label assigned by the linear probe model (e.g., `class_A`).

#### Usage
```bash
python inference_cls.py <feature_folder> <output_file>
```
- `feature_folder`: Directory containing input feature files (in `.npy` format).
- `output_file`: File path to save the classification results (in JSON format).

### 3. `train_cls.py`
This script is designed for training the linear classification model.

#### Usage
```bash
python train_cls.py
```

### 4. `utils.py`
The utility script defines the architecture of the linear classification model.

## Naming Convention
All `.npy` files used in this codebase must follow the naming convention of `label_filename.npy`, where the filename should not contain any underscores (`_`).
