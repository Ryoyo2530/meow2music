import os

task = "move files"

if task == "creat project" :
    folders = [
        "data/train_dataset/sample_01",
        "vocalnet",
        "colab",
        "utils",
        "checkpoints"
    ]

    files = {
        "vocalnet/__init__.py": "",
        "vocalnet/model.py": "# VocalNet model will go here\n",
        "vocalnet/dataset.py": "# Dataset class\n",
        "vocalnet/loss.py": "# Loss functions\n",
        "vocalnet/train.py": "# Training script\n",
        "colab/train_colab.ipynb": "# Colab notebook placeholder\n",
        "utils/audio_utils.py": "# Audio utility functions\n",
        "requirements.txt": "torch\ntorchaudio\nlibrosa\nnumpy\nscipy\ntqdm\n",
        "README.md": "# Meow2Music Project\n\nGenerate cat vocals over instrumentals.",
        # ".gitignore": "__pycache__/\n*.pt\n*.wav\ncheckpoints/\ndata/\n"
    }

    # Create folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    # Create sample folders
    for idx in range(10):
        os.makedirs(f"data/train_dataset/sample_{idx}", exist_ok=True)

    # Create files with content
    for path, content in files.items():
        with open(path, 'w') as f:
            f.write(content)

    print("✅ Meow2Music project structure created.")


import os
import shutil

if task == "move files":
    source_dir = "data/train_dataset/"  # Directory containing the .wav files
    destination_base = "data/train_dataset/"  # Base directory for sample folders

    # List all .wav files in the source directory
    wav_files = [f for f in os.listdir(source_dir) if f.endswith(".wav")]

    # Process each file
    for filename in wav_files:
        # Extract idx from the filename (e.g., filename_0.wav → idx = 0)
        idx = filename.split("_")[-1].split(".")[0]
        
        # Define source and destination paths
        source_path = os.path.join(source_dir, filename)
        destination_folder = os.path.join(destination_base, f"sample_{idx}")
        destination_path = os.path.join(destination_folder, filename.replace(f"_{idx}", ""))
        
        # Ensure the destination folder exists
        os.makedirs(destination_folder, exist_ok=True)
        
        # Move and rename the file
        shutil.move(source_path, destination_path)

    print("✅ Files moved and renamed successfully.")