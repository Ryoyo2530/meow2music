import torch
import torch.nn as nn

class SpectrogramLoss(nn.Module):
    def __init__(self, l1_weight=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchcrepe

# Constants
SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 512

# Mel transform
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

def compute_mel(x):
    """Mel spectrogram of audio."""
    return mel_transform(x)

def compute_f0(x):
    """Estimate pitch using torchcrepe."""
    # x: [B, T] or [T], torchcrepe expects mono
    if x.ndim == 2:
        x = x.squeeze(0)
    x = x.unsqueeze(0)  # [1, T]
    f0, pd = torchcrepe.predict(
        audio=x,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=50.0,
        fmax=1100.0,
        model='full',         # or 'tiny'
        batch_size=512,
        return_periodicity=True,
        device=x.device,
        pad=True
    )
    return f0  # [1, T'] pitch contour

def compute_onset_strength(x):
    """Rough rhythm/onset estimation using energy deltas."""
    # Compute short-time energy
    mel = compute_mel(x)  # [B, N_MELS, T]
    energy = mel.mean(dim=1)  # [B, T]
    delta = torchaudio.functional.compute_deltas(energy)  # [B, T]
    onset_strength = F.relu(delta)  # [B, T]
    return onset_strength

def loss_fn(pred, meow, instrument,
            λ_mel=1.0, λ_pitch=1.0, λ_rhythm=1.0):
    """Total loss for VocalNet generation.

    Arguments:
        pred: generated waveform [B, T]
        meow: original meow waveform [B, T]
        instrument: instrumental audio [B, T]
    Returns:
        total loss
    """
    # Mel identity loss: preserve vocal texture
    mel_pred = compute_mel(pred)
    mel_meow = compute_mel(meow)
    mel_loss = F.l1_loss(mel_pred, mel_meow)

    # Pitch loss: follow the instrument melody
    f0_pred = compute_f0(pred)
    f0_inst = compute_f0(instrument)
    pitch_loss = F.l1_loss(f0_pred, f0_inst)

    # Rhythm loss: align onset strength
    onset_pred = compute_onset_strength(pred)
    onset_inst = compute_onset_strength(instrument)
    rhythm_loss = F.mse_loss(onset_pred, onset_inst)

    total = λ_mel * mel_loss + λ_pitch * pitch_loss + λ_rhythm * rhythm_loss
    return total, {
        'mel_loss': mel_loss.item(),
        'pitch_loss': pitch_loss.item(),
        'rhythm_loss': rhythm_loss.item(),
        'total': total.item()
    }

    def forward(self, pred, target):
        """
        pred, target: (B, T, F) mel spectrograms
        """
        l1 = self.l1(pred, target)
        return self.l1_weight * l1
    

