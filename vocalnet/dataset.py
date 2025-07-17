from torch.utils.data import Dataset
import torchaudio
import os

class MeowVocalDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000):
        self.sample_rate = sample_rate
        self.pairs = []
        for folder in os.listdir(root_dir):
            pair = (
                os.path.join(root_dir, folder, 'meow.wav'),
                os.path.join(root_dir, folder, 'instrument.wav')
            )
            self.pairs.append(pair)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        meow_path, instr_path = self.pairs[idx]
        meow, _ = torchaudio.load(meow_path)
        instr, _ = torchaudio.load(instr_path)
        
        # Convert stereo to mono, ensure channel=1
        meow = meow.mean(dim=0, keepdim=True) if meow.shape[0] > 1 else meow
        instr = instr.mean(dim=0, keepdim=True) if instr.shape[0] > 1 else instr
                
        # Pad or crop to same length
        min_len = min(meow.shape[1], instr.shape[1])
        meow = meow[:, :min_len]
        instr = instr[:, :min_len]

        return meow, instr
