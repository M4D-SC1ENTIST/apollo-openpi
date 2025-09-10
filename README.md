This was modified based on openpi.

# Installation
- Clone the repo
- Set up the environment with uv
    ```bash
    GIT_LFS_SKIP_SMUDGE=1 uv sync
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
    ```

# Dataset Conversion
- Download the dataset
    ```
    uv run prepare_dataset.py
    ```
- Convert the raw dataset from ZKTP
    ```bash
    uv run zktp/convert_zktp_data_to_lerobot.py --data_dir datasets/raw_data
    ```
    
- Check if the dataset looks correct by randomly sampling one episode
    ```bash
    uv run zktp/play_random_episode.py
    ```
 
 # Training
 - pi0 fine-tuning
    ```
    uv run scripts/compute_norm_stats.py --config-name pi0_zktp

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_zktp --exp-name=zktp-pi0 --overwrite
    ```

- pi0-fast fine-tuning
    ```
    uv run scripts/compute_norm_stats.py --config-name pi0_fast_zktp

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_zktp --exp-name=zktp-pi0-fast --overwrite

    ```

- pi0.5 fine-tuning
    ```
    uv run scripts/compute_norm_stats.py --config-name pi05_zktp

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_zktp --exp-name=zktp-pi05 --overwrite
    ```

