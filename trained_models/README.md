# Model Files

⚠️ **Model files not included in repository** ⚠️

The trained model `.pkl` files are **not pushed to GitHub** due to size constraints (>100 MB each).

## Trained Models (Local Only):

### Main Demo Model
- **File**: `trained_models/demo_ready_snn_model.pkl` (347 MB)
- **Accuracy**: 97.05%
- **Metadata**: `trained_models/demo_ready_snn_metadata.json` (included ✅)
- **Architecture**: 512→384→256→10 LIF neurons

### Other Models
- `trained_models/gpu_snn_final_model.pkl` - Final GPU SNN model
- `source_code/models/advanced_neuromorphic_detector.pkl` (265 MB) - Advanced model
- `models/improved_sound_detector.pkl` - Improved detector

## To Use This Project:

1. **Train the model yourself** using:
   ```bash
   python scripts/final_training/train_gpu_snn_final.py
   ```

2. **Or download pre-trained models** from:
   - [Add your preferred hosting service like Google Drive, Hugging Face, etc.]

3. **Model metadata files are included** (JSON files with architecture details)

## Model Information:

The metadata JSON files contain all the information needed to recreate the models:
- Architecture specifications
- Training parameters
- Feature extraction settings
- Performance metrics

All the **training scripts** and **source code** are included in this repository to reproduce the 97% accuracy model.
