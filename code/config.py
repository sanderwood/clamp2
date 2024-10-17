EVAL_SPLIT = 0.01  # Fraction of training data used for evaluation
WANDB_KEY = "<your_wandb_key>"  # Set M3/CLaMP2_WANDB_LOG=False if no API key for Weights and Biases logging

# -------------------- Configuration for M3 Training --------------------
TRAIN_FOLDERS = [
    "<path_to_training_data>"  # Directory containing training data
]

EVAL_FOLDERS = [
    ""  # (Optional) Directory containing evaluation data
]

PATCH_SIZE = 64  # Size of each patch
PATCH_LENGTH = 512  # Length of the patches
PATCH_NUM_LAYERS = 12  # Number of layers in the encoder
TOKEN_NUM_LAYERS = 3  # Number of layers in the decoder
M3_HIDDEN_SIZE = 768  # Size of the hidden layer

M3_NUM_EPOCH = 100  # Maximum number of epochs for training
M3_LEARNING_RATE = 1e-4  # Learning rate for the optimizer
M3_BATCH_SIZE = 16  # Batch size per GPU (single card) during training
M3_MASK_RATIO = 0.45  # Ratio of masked elements during training
M3_DETERMINISTIC = True  # Ensures deterministic results with random seeds
M3_WANDB_LOG = True  # Enable logging to Weights and Biases
M3_LOAD_CKPT = True  # Load model weights from a checkpoint if available

M3_WEIGHTS_PATH = (
    "weights_m3_p_size_" + str(PATCH_SIZE) +
    "_p_length_" + str(PATCH_LENGTH) +
    "_t_layers_" + str(TOKEN_NUM_LAYERS) +
    "_p_layers_" + str(PATCH_NUM_LAYERS) +
    "_h_size_" + str(M3_HIDDEN_SIZE) +
    "_lr_" + str(M3_LEARNING_RATE) +
    "_batch_" + str(M3_BATCH_SIZE) +
    "_mask_" + str(M3_MASK_RATIO) + ".pth"
)  # Path to store the model weights
M3_LOGS_PATH = M3_WEIGHTS_PATH.replace("weights", "logs").replace("pth", "txt")  # Path to save training logs

# -------------------- Configuration for CLaMP2 Training ----------------
TRAIN_JSONL = "<path_to_training_jsonl>"  # Path to the JSONL file with training data
EVAL_JSONL = ""  # (Optional) Path to the JSONL file with evaluation data

CLAMP2_HIDDEN_SIZE = 768  # Size of the hidden layer
TEXT_MODEL_NAME = "FacebookAI/xlm-roberta-base"  # Name of the pre-trained text model

CLAMP2_NUM_EPOCH = 100  # Maximum number of epochs for training
CLAMP2_LEARNING_RATE = 5e-5  # Learning rate for the optimizer
CLAMP2_BATCH_SIZE = 128  # Batch size per GPU (single card) during training
LOGIT_SCALE = 1  # Scaling factor for contrastive loss
MAX_TEXT_LENGTH = 128  # Maximum allowed length for text input
TEXT_DROPOUT = True  # Whether to apply dropout during text processing
CLAMP2_DETERMINISTIC = True  # Ensures deterministic results with random seeds
CLAMP2_LOAD_M3 = True  # Load weights from the M3 model
CLAMP2_WANDB_LOG = True  # Enable logging to Weights and Biases
CLAMP2_LOAD_CKPT = True  # Load weights from a checkpoint if available

CLAMP2_WEIGHTS_PATH = (
    "weights_clamp2_h_size_" + str(CLAMP2_HIDDEN_SIZE) +
    "_lr_" + str(CLAMP2_LEARNING_RATE) +
    "_batch_" + str(CLAMP2_BATCH_SIZE) +
    "_scale_" + str(LOGIT_SCALE) +
    "_t_length_" + str(MAX_TEXT_LENGTH) +
    "_t_model_" + TEXT_MODEL_NAME.replace("/", "_") +
    "_t_dropout_" + str(TEXT_DROPOUT) +
    "_m3_" + str(CLAMP2_LOAD_M3) + ".pth"
)  # Path to store CLaMP2 model weights
CLAMP2_LOGS_PATH = CLAMP2_WEIGHTS_PATH.replace("weights", "logs").replace("pth", "txt")  # Path to save training logs
