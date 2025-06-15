import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel
import torch.nn.functional as F
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

class TransformerProjector(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=768, num_tokens=2304, out_seq_len=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(num_tokens, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.norm = nn.LayerNorm(hidden_dim)
        self.to_seq = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.output_proj = nn.Linear(num_tokens, out_seq_len)

    def forward(self, x):  # x: [B, N, D]
        B, N, _ = x.shape
        pos = self.pos_embed[:N].unsqueeze(0)  # support dynamic length
        x = self.proj(x) + pos                 # [B, N, 768]
        x = x.transpose(0, 1)                  # [N, B, 768]
        x = self.encoder(x)                    # [N, B, 768]
        x = self.norm(x.transpose(0, 1))       # [B, N, 768]
        x = self.to_seq(x.transpose(1, 2))     # [B, 768, N]
        x = self.output_proj(x)                # [B, 768, seq_len]
        x = x.transpose(1, 2)                  # [B, seq_len, 768]
        return x


def generate_caption(projector, ply_feature):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t5 = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    projector.eval()
    t5.eval()

    x = ply_feature.unsqueeze(0).to(device)  # [1, N, D]
    with torch.no_grad():
        encoder_outputs = projector(x)  # [1, seq_len, 768]
        output_ids = t5.generate(encoder_outputs=encoder_outputs, max_length=64)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption
