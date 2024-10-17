import os
import torch
import numpy as np
import argparse

def get_features(path):
    """
    Load and return feature data from .npy files in the given directory.
    Each feature is stored in a dictionary with the filename (without extension) as the key.
    """
    files = sorted(os.listdir(path))
    features = {}
    
    for file in files:
        if file.endswith(".npy"):
            key = file.split(".")[0]
            features[key] = np.load(os.path.join(path, file))[0]
    
    return features

def calculate_metrics(query_features, reference_features):
    """
    Calculate MRR, Hit@1, Hit@10, and Hit@100 metrics based on the similarity 
    between query and reference features.
    """
    common_keys = set(query_features.keys()) & set(reference_features.keys())
    mrr, hit_1, hit_10, hit_100 = 0, 0, 0, 0

    for idx, key in enumerate(common_keys):
        # Convert query feature to tensor and add batch dimension
        query_feat = torch.tensor(query_features[key]).unsqueeze(dim=0)
        
        # Collect all reference features for common keys
        ref_feats = torch.tensor(np.array([reference_features[k] for k in common_keys]))

        # Compute cosine similarity between the query and all reference features
        similarities = torch.cosine_similarity(query_feat, ref_feats)

        # Create a list of (similarity, index) pairs
        indexed_sims = list(enumerate(similarities.tolist()))

        # Sort by similarity in descending order, with idx-based tie-breaking
        sorted_indices = sorted(indexed_sims, key=lambda x: (x[1], x[0] == idx), reverse=True)

        # Extract the sorted rank list
        ranks = [x[0] for x in sorted_indices]

        # Calculate MRR
        mrr += 1 / (ranks.index(idx) + 1)

        # Calculate Hit@1, Hit@10, Hit@100
        if idx in ranks[:100]:
            hit_100 += 1
            if idx in ranks[:10]:
                hit_10 += 1
                if idx in ranks[:1]:
                    hit_1 += 1

    # Compute the final metrics
    total_keys = len(common_keys)
    print(f"MRR: {round(mrr / total_keys, 4)}")
    print(f"Hit@1: {round(hit_1 / total_keys, 4)}")
    print(f"Hit@10: {round(hit_10 / total_keys, 4)}")
    print(f"Hit@100: {round(hit_100 / total_keys, 4)}")

if __name__ == '__main__':
    # Set up argument parsing for input directories
    parser = argparse.ArgumentParser(description="Calculate similarity metrics between query and reference features.")
    parser.add_argument('query_folder', type=str, help='Path to the folder containing query features (.npy files).')
    parser.add_argument('reference_folder', type=str, help='Path to the folder containing reference features (.npy files).')
    args = parser.parse_args()

    # Load features from the specified folders
    query_features = get_features(args.query_folder)
    reference_features = get_features(args.reference_folder)

    # Calculate and print the metrics
    calculate_metrics(query_features, reference_features)
