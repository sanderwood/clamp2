# Configuration for generative modelling and classification
TRAIN_FOLDERS = [
    "<path_to_training_data>"  # Directory containing training data
]

EVAL_FOLDERS = [
    ""  # (Optional) Directory containing evaluation data
]

EVAL_SPLIT = 0.2  # Fraction of training data to use for evaluation

# Weights and Biases configuration
WANDB_KEY = "<your_wandb_key>"  # Set M3/CLaMP2_WANDB_LOG=False if no API key for Weights and Biases logging

# Model Configuration
INPUT_HIDDEN_SIZE = 768  # Input hidden size
HIDDEN_SIZE = 768  # Model hidden size
NUM_EPOCHS = 1000  # Max number of epochs to train (early stopping can terminate earlier)
LEARNING_RATE = 1e-5  # Optimizer learning rate
BALANCED_TRAINING = False  # Set to True to balance labels in training data
WANDB_LOG = False  # Set to True to log training metrics to WANDB

# Paths Configuration
last_folder_name = TRAIN_FOLDERS[-1].split('/')[-1]
WEIGHTS_PATH = f"weights-{last_folder_name}.pth"  # Weights file path
LOGS_PATH = f"logs-{last_folder_name}.txt"  # Log file path
