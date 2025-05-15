import os
import gc
import cv2
import time
import math
import timm
import torch
import librosa
import logging
import warnings
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

class CFG:
    # Mel Spec
    # -------------------------------------------
    N_FFT = 1024
    WIN_LENGTH = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 20
    FMAX = 16000
    TARGET_SHAPE = (256, 256)
    FS = 32000
    WINDOW_SIZE = 5
    # Model
    # -------------------------------------------
    model_path = '/kaggle/input/efficientnetv2_s_auc0.98/pytorch/default/1'
    model_name = 'tf_efficientnetv2_s.in21k_ft_in1k'
    use_specific_folds = True
    folds = [0, 1, 2, 3]
    in_channels = 1
    device = 'cpu'
    # Datasets / Paths
    # -------------------------------------------
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    # Hyper Param
    # -------------------------------------------
    batch_size = 32
    use_tta = False
    tta_count = 3
    threshold = 0.7
    # Utils
    # -------------------------------------------
    debug = False
    debug_count = 3

cfg = CFG()
taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
species_ids = taxonomy_df['primary_label'].tolist()
num_classes = len(species_ids)

class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,
            in_chans=cfg.in_channels,
            drop_rate=0.0,
            drop_path_rate=0.0
        )
        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name or 'resnext' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif 'edgenext' in cfg.model_name:
            backbone_out = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
        elif 'convnext' in cfg.model_name:
            backbone_out = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, dict):
            features = features['features']
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits

def audio2melspec(audio_data, cfg):
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0,
        pad_mode="reflect",
        norm='slaney',
        htk=True,
        center=True,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_spec_norm

def process_audio_segment(audio_data, cfg):
    if len(audio_data) < cfg.FS * cfg.WINDOW_SIZE:
        audio_data = np.pad(audio_data,
                            (0, cfg.FS * cfg.WINDOW_SIZE - len(audio_data)),
                            mode='constant')
    mel_spec = audio2melspec(audio_data, cfg)
    if mel_spec.shape != cfg.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
    return mel_spec.astype(np.float32)

def find_model_files(cfg):
    model_files = []
    model_dir = Path(cfg.model_path)
    for path in model_dir.glob('**/*.pth'):
        model_files.append(str(path))
    return model_files

def load_models(cfg, num_classes):
    models = []
    model_files = find_model_files(cfg)
    if not model_files:
        print(f"Warning: No model files found under {cfg.model_path}!")
        return models
    print(f"Found a total of {len(model_files)} model files.")
    if cfg.use_specific_folds:
        filtered_files = []
        for fold in cfg.folds:
            fold_files = [f for f in model_files if f"_{fold}" in f]
            filtered_files.extend(fold_files)
        model_files = filtered_files
        print(f"Using {len(model_files)} model files for the specified folds ({cfg.folds}).")
    for model_path in model_files:
        try:
            print(f"Loading model: {model_path}")
            checkpoint = torch.load(model_path, map_location=torch.device(cfg.device))
            model = BirdCLEFModel(cfg, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(cfg.device)
            model.eval()
            models.append(model)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
    return models

def predict_on_spectrogram(audio_path, models, cfg, species_ids):
    predictions = []
    row_ids = []
    soundscape_id = Path(audio_path).stem
    try:
        print(f"Processing {soundscape_id}")
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        total_segments = int(len(audio_data) / (cfg.FS * cfg.WINDOW_SIZE))
        for segment_idx in range(total_segments):
            start_sample = segment_idx * cfg.FS * cfg.WINDOW_SIZE
            end_sample = start_sample + cfg.FS * cfg.WINDOW_SIZE
            segment_audio = audio_data[start_sample:end_sample]
            end_time_sec = (segment_idx + 1) * cfg.WINDOW_SIZE
            row_id = f"{soundscape_id}_{end_time_sec}"
            row_ids.append(row_id)
            if cfg.use_tta:
                all_preds = []
                for tta_idx in range(cfg.tta_count):
                    mel_spec = process_audio_segment(segment_audio, cfg)
                    mel_spec = apply_tta(mel_spec, tta_idx)
                    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    mel_spec = mel_spec.to(cfg.device)
                    if len(models) == 1:
                        with torch.no_grad():
                            outputs = models[0](mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            all_preds.append(probs)
                    else:
                        segment_preds = []
                        for model in models:
                            with torch.no_grad():
                                outputs = model(mel_spec)
                                probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                                segment_preds.append(probs)
                        avg_preds = np.mean(segment_preds, axis=0)
                        all_preds.append(avg_preds)
                final_preds = np.mean(all_preds, axis=0)
            else:
                mel_spec = process_audio_segment(segment_audio, cfg)
                mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mel_spec = mel_spec.to(cfg.device)
                if len(models) == 1:
                    with torch.no_grad():
                        outputs = models[0](mel_spec)
                        final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
                else:
                    segment_preds = []
                    for model in models:
                        with torch.no_grad():
                            outputs = model(mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            segment_preds.append(probs)
                    final_preds = np.mean(segment_preds, axis=0)
            predictions.append(final_preds)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
    return row_ids, predictions

def apply_tta(spec, tta_idx):
    if tta_idx == 0:
        return spec
    elif tta_idx == 1:
        return np.flip(spec, axis=1)
    elif tta_idx == 2:
        return np.flip(spec, axis=0)
    else:
        return spec

def run_inference(cfg, models, species_ids):
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))
    if cfg.debug:
        print(f"Debug mode enabled, using only {cfg.debug_count} files")
        test_files = test_files[:cfg.debug_count]
    print(f"Found {len(test_files)} test soundscapes")
    all_row_ids = []
    all_predictions = []
    for audio_path in tqdm(test_files):
        row_ids, predictions = predict_on_spectrogram(str(audio_path), models, cfg, species_ids)
        all_row_ids.extend(row_ids)
        all_predictions.extend(predictions)
    return all_row_ids, all_predictions

def create_submission(row_ids, predictions, species_ids, cfg):
    print("Creating submission dataframe...")
    submission_dict = {'row_id': row_ids}
    for i, species in enumerate(species_ids):
        submission_dict[species] = [pred[i] for pred in predictions]
    submission_df = pd.DataFrame(submission_dict)
    submission_df.set_index('row_id', inplace=True)
    sample_sub = pd.read_csv(cfg.submission_csv, index_col='row_id')
    missing_cols = set(sample_sub.columns) - set(submission_df.columns)
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} species columns in submission")
        for col in missing_cols:
            submission_df[col] = 0.0
    submission_df = submission_df[sample_sub.columns]
    submission_df = submission_df.reset_index()
    return submission_df

def smooth_submission():
    sub = pd.read_csv('submission.csv')
    cols = sub.columns[1:]
    groups = sub['row_id'].str.rsplit('_', n=1).str[0]
    groups = groups.values
    for group in np.unique(groups):
        sub_group = sub[group == groups]
        predictions = sub_group[cols].values
        new_predictions = predictions.copy()
        for i in range(1, predictions.shape[0]-1):
            new_predictions[i] = (predictions[i-1] * 0.2) + (predictions[i] * 0.6) + (predictions[i+1] * 0.2)
        new_predictions[0] = (predictions[0] * 0.9) + (predictions[1] * 0.1)
        new_predictions[-1] = (predictions[-1] * 0.9) + (predictions[-2] * 0.1)
        sub_group[cols] = new_predictions
        sub[group == groups] = sub_group
    sub.to_csv("submission.csv", index=False)

def main():
    start_time = time.time()
    print(f"TTA enabled: {cfg.use_tta} (variations: {cfg.tta_count if cfg.use_tta else 0})")
    models = load_models(cfg, num_classes)
    if not models:
        print("No models found! Please check model paths.")
        return
    print(f"Model usage: {'Single model' if len(models) == 1 else f'Ensemble of {len(models)} models'}")
    row_ids, predictions = run_inference(cfg, models, species_ids)
    submission_df = create_submission(row_ids, predictions, species_ids, cfg)
    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    end_time = time.time()
    smooth_submission()
    print(f"Inference completed in {(end_time - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()