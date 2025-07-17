import torch
import torch.nn as nn

# cnn-based vocalnet
class VocalNet_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(128*2, 128, kernel_size=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, meow, instr):
        x1 = self.encoder(meow)
        x2 = self.encoder(instr)
        x = self.fusion(torch.cat([x1, x2], dim=1))
        return self.decoder(x)

# transformer-based vocalnet
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class VocalNet_Transformer(nn.Module):
    def __init__(self, input_dim=80, embed_dim=256, nhead=4, num_layers=4, output_dim=80):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=512, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, meow_feats, instrument_feats=None):
        """
        Inputs:
          meow_feats: (B, T, F) - raw meow mel features
          instrument_feats: (B, T, F) - optionally include conditioning features from instrument
        """
        x = self.input_proj(meow_feats)  # (B, T, embed_dim)
        x = self.pos_encoder(x)

        x = x.permute(1, 0, 2)  # (T, B, embed_dim) for transformer
        x = self.encoder(x)     # (T, B, embed_dim)
        x = x.permute(1, 0, 2)  # (B, T, embed_dim)

        out = self.output_proj(x)  # (B, T, output_dim)
        return out
