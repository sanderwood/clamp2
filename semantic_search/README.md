# Semantic Search Codebase

## Overview
CLaMP 2 is a state-of-the-art multimodal music information retrieval system designed to work with 101 languages. This codebase includes scripts for evaluating model performance, performing semantic searches, and calculating similarity metrics based on CLaMP2-extracted **nomarlized** feature vectors from music or text data. Below is a description of the scripts contained in the `semantic_search/` folder.

## Repository Structure
The `semantic_search/` folder contains the following scripts:

### 1. `clamp2_score.py`
This script calculates the cosine similarity between the average feature vectors extracted from two sets of `.npy` files, serving as a measure of similarity between the reference and test datasets.

It can be used to validate the semantic similarity between generated music and ground truth, providing an objective metric. Through empirical observation, we found that this metric aligns well with subjective judgments made by individuals with professional music expertise.

**Usage:**
```bash
python clamp2_score.py <reference_folder> <test_folder>
```
- `reference_folder`: Path to the folder containing reference `.npy` files.
- `test_folder`: Path to the folder containing test `.npy` files.

**Functionality:**
- Loads all `.npy` files from the specified folders.
- Computes the average feature vector for each folder.
- Calculates the cosine similarity between the two averaged vectors.
- Outputs the similarity score rounded to four decimal places.

### 2. `semantic_search.py`
This script performs semantic search by calculating the cosine similarity between a query feature and a set of features stored in `.npy` files.

**Usage:**
```bash
python semantic_search.py <query_file> <features_folder> [--top_k TOP_K]
```
- `query_file`: Path to the query feature file (e.g., `ballad.npy`).
- `features_folder`: Path to the folder containing feature files for comparison.
- `--top_k`: (Optional) Number of top similar items to display. Defaults to 10 if not specified.

**Functionality:**
- Loads a query feature from the specified file.
- Loads feature vectors from the given folder.
- Computes cosine similarity between the query feature and each loaded feature vector.
- Displays the top K most similar features along with their similarity scores.

### 3. `semantic_search_metrics.py`
This script calculates evaluation metrics for semantic search by comparing query features to reference features.

**Usage:**
```bash
python semantic_search_metrics.py <query_folder> <reference_folder>
```
- `query_folder`: Path to the folder containing query features (in `.npy` format).
- `reference_folder`: Path to the folder containing reference features (in `.npy` format).

**Functionality:**
- Loads query features from the specified folder.
- Loads reference features from the given folder.
- Computes the following metrics based on cosine similarity:
  - **Mean Reciprocal Rank (MRR)**
  - **Hit@1**
  - **Hit@10**
  - **Hit@100**
- Outputs the calculated metrics to the console.
