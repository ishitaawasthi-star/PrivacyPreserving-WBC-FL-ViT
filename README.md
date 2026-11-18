# Privacy-Preserving WBC Diagnosis — Federated ViT

A runnable research prototype based on the uploaded paper **“Privacy-Preserving Medical Image Diagnosis using Federated Learning and Vision Transformers.”**

It supports:
- Four WBC classes: `eosinophil`, `lymphocyte`, `monocyte`, `neutrophil`
- 32×32 image preprocessing
- A compact Vision Transformer (ViT)
- Three simulated clients trained locally with Federated Averaging (FedAvg)
- Optional differentially-private local training using Opacus (experimental)
- A Streamlit interface for training, metrics, and single-image inference

> **Important:** This is an academic prototype, not a clinically validated diagnostic system. Do not use its predictions for patient care. The paper reports preliminary results (centralized ViT 58.50% accuracy; federated ViT 49.50%), and this code does not claim to reproduce those exact results. Results depend on your dataset, split, hardware, and training settings.

## 1. Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10 or 3.11 recommended
- VS Code
- Internet connection for the first installation only

## 2. Setup in VS Code

Open this folder in VS Code, then open Terminal.

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### macOS / Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will show a local URL, usually `http://localhost:8501`.

## 3. Add your dataset

The project does not bundle the paper's Blood Cell Images dataset. Place your images in this structure:

```text
data/
└── blood_cells/
    ├── eosinophil/
    │   ├── image001.jpg
    │   └── ...
    ├── lymphocyte/
    ├── monocyte/
    └── neutrophil/
```

Accepted formats: JPG, JPEG, PNG, BMP. Each class folder must contain images. Use a balanced, correctly labelled dataset where possible. The app makes a stratified train/validation split.

## 4. Train

In the app:
1. Select **Training**.
2. Set epochs, communication rounds, and local epochs.
3. Choose **Federated ViT (3 clients)**.
4. Click **Start training**.
5. The global checkpoint and metrics are saved under `models/`.

The default setup is intentionally small for a CPU-friendly demonstration. For meaningful research, use a larger dataset and tune the settings.

## 5. Predict

After training, open **Prediction**, choose the saved global ViT checkpoint, and upload a WBC image. The app displays the predicted class and class probabilities.

## 6. Optional command-line training

```bash
python -m src.train --data_dir data/blood_cells --rounds 5 --local_epochs 1 --batch_size 16
```

For centralized ViT:
```bash
python -m src.train --data_dir data/blood_cells --mode centralized --epochs 10
```

## 7. Privacy notes

- FedAvg simulates three clients in one process. It demonstrates the training workflow; it is not a deployed multi-hospital network.
- Raw images are loaded by the local training process and are not transmitted between simulated clients. Only model state dictionaries are aggregated in this simulation.
- Differential privacy is optional and requires Opacus. It clips per-sample gradients and adds noise during local training. The app does **not** claim a formal privacy guarantee or display an epsilon value unless a correct accountant is implemented for the exact training configuration.
- Secure aggregation, encrypted transport, access control, and real client isolation are not implemented.
- Do not upload identifiable patient data to an untrusted environment.

## 8. GitHub

Before publishing:
1. Add your dataset only if its licence permits redistribution. Usually keep medical datasets out of Git.
2. Do not commit patient-identifiable images or trained weights derived from restricted data.
3. Review the dataset licence and add the required attribution.
4. Commit the code:
```bash
git init
git add .
git commit -m "Add federated ViT WBC research prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

## Project layout

```text
app.py                  Streamlit UI
src/
  data.py               Dataset loading and transforms
  models.py             Compact ViT and CNN baseline
  federated.py          FedAvg client training
  train.py              Training CLI
  utils.py              Metrics and checkpoint helpers
scripts/
  check_environment.py  Installation check
data/blood_cells/       Put the four class folders here
models/                 Checkpoints and metrics are saved here
```
