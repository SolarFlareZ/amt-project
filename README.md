# Automatic Music Transcription (AMT)

Deep learning system for automatic piano transcription from audio to symbolic notation (MIDI), using the MAESTRO dataset.

## Deliverables

| File | Description |
|---|---|
| `report.pdf` | Project report |
| `Experiment spreadsheet.xlsx` | All training statistics, metrics, and W&B training logs |
| `Self-assessment.docx` | Self-assessment document |

## Installation

```bash
git clone https://github.com/SolarFlareZ/amt-project.git
cd amt-project
pip install -e .
```

### Other dependencies
For some PyTorch versions, FFmpeg is required.

## Usage

### 1. Preprocessing

Convert audio and MIDI files to cached spectrograms:

```bash
# MAESTRO dataset
python3 scripts/preprocess.py dataset=maestro paths.dataset_dir=/path/to/maestro-v3.0.0 audio=mel

# MAPS dataset (for cross-dataset evaluation)
python3 scripts/preprocess.py dataset=maps paths.dataset_dir=/path/to/MAPS audio=mel
```

Supported spectrogram types: `mel`, `cqt`, `stft`

### 2. Training

```bash
# Train CNN
python3 scripts/train.py audio=mel model=cnn

# Train CRNN
python3 scripts/train.py audio=cqt model=crnn

# Train CRNN with frozen CNN backbone
python3 scripts/train.py audio=mel model=crnn model.pretrained_cnn_path=./checkpoints/best_cnn.ckpt model.freeze_backbone=true
```

### 3. Hyperparameter Tuning

```bash
python3 scripts/tune.py audio=mel model=cnn
python3 scripts/tune.py audio=cqt model=crnn model.pretrained_cnn_path=./checkpoints/best_cnn.ckpt model.freeze_backbone=true
```

### 4. Evaluation

```bash
# Evaluate on MAESTRO test set
python3 scripts/evaluate.py audio=mel paths.checkpoint_path=./checkpoints/best.ckpt

# Evaluate on MAPS (cross-dataset)
python3 scripts/evaluate_maps.py audio=mel paths.checkpoint_path=./checkpoints/best.ckpt +threshold=0.55

# Robustness evaluation
python3 scripts/evaluate_robustness.py audio=mel paths.checkpoint_path=./checkpoints/best.ckpt
```

## Datasets

- [**MAESTRO v3**](https://magenta.tensorflow.org/datasets/maestro): Training, validation, and testing.
- [**MAPS**](https://adasp.telecom-paris.fr/resources/2010-07-08-maps-database/): Cross-dataset evaluation.

## Trained Models

All trained model checkpoints are available at:
[Google Drive](https://drive.google.com/drive/folders/1GGAUY34ktsGUQDCQ7I6JeLXrAbsvp5Se?usp=drive_link)