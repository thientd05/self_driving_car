# Self Driving Car

End-to-end pipeline for training a behavioral cloning model and driving the
Udacity-style self-driving car simulator via Socket.IO.

### Video demo
- demo.mp4
or
- Demo link: https://youtube.com/shorts/abeWI6rHyNw?feature=share

## Project Structure 

- `src/main/model.py` – CNN architecture that maps images to steering & throttle.
- `src/main/utils.py` – `keras.utils.Sequence` for loading, balancing, and augmenting driving logs.
- `src/main/train_model.py` – CLI for training with checkpointing, LR scheduling, and early stopping.
- `src/main/control.py` – Socket.IO controller that streams predictions to the simulator.
- `weights/` – Pretrained `.keras` checkpoints and reference screenshots.
- `explain_model.md` – Notes on model decisions and experiments.

## Environment Setup

1. (Recommended) create a virtualenv:
   ```bash
   cd ~your_dir # clone repo first
   python3 -m venv env
   source env/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Preparing Data

1. Record driving data from the simulator to produce `driving_log.csv` and the
   associated `IMG/` folder.
2. Split the log (and corresponding images) into training and validation CSVs.
3. Ensure the CSV columns follow the standard Udacity format:
   `center,left,right,steering,throttle,brake,speed`.

## Training

```bash
cd ~your_dir/src/main
python train_model.py \
  --train_csv_file /path/to/train.csv \
  --val_csv_file /path/to/val.csv \
  --batchsize 32 \
  --epochs 30
```

Outputs are saved under `models5/` by default (best checkpoints only).

## Driving the Simulator

1. Launch the simulator in autonomous mode.
2. Start the controller:
   ```bash
   cd ~your_dir/src/main
   python control.py weights/save_at21.keras
   ```
3. On connection, the script streams steering/throttle commands predicted from
   incoming base64 images.

## Tips & Notes

- `utils.dataset` performs balancing (down-sampling zero steering) and simple
  augmentations (flips, brightness).
- Throttle labels are mapped to `[-1, 1]` internally; the controller remaps to
  `[0, 1]` throttle output.
- Adjust `models5/` or callback settings in `train_model.py` to match your
  storage or experiment tracking needs.
- See `explain_model.md` for architecture rationale and future work ideas.

