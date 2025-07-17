import torch
import torchaudio
import os
import argparse
from vocalnet.model import VocalNet_CNN
import soundfile as sf

def load_model(checkpoint_path, device):
    model = VocalNet_CNN()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval().to(device)
    return model

def load_wav(path, sample_rate, max_len=None):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] == 2:
        wav = wav.mean(dim=0, keepdim=True)  # stereo → mono
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if max_len is not None:
        if wav.shape[1] > max_len:
            wav = wav[:, :max_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, max_len - wav.shape[1]))
    return wav

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    meow = load_wav(args.meow, args.sample_rate, args.max_len).to(device)   # (1, T)
    instr = load_wav(args.instr, args.sample_rate, args.max_len).to(device) # (1, T)

    meow = meow.unsqueeze(0)   # (B, 1, T)
    instr = instr.unsqueeze(0) # (B, 1, T)

    with torch.no_grad():
        output = model(meow, instr)  # (B, T, F)
        output = output.squeeze(0)
    
    waveform = output.squeeze(0).cpu().numpy()
    print(waveform.shape)
    print(waveform)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "generated_1.wav")
    
    sf.write(out_path, waveform, samplerate=16000)
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/vocalnet_epoch10.pt")
    parser.add_argument("--meow", type=str, default="data/train_dataset/sample_1/meow.wav", help="Path to meow .wav")
    parser.add_argument("--instr", type=str, default="data/train_dataset/sample_1/instrument.wav", help="Path to instrument .wav")
    parser.add_argument("--out_dir", type=str, default="data/inference_output")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_len", type=int, default=10*16000)

    args = parser.parse_args()
    run_inference(args)