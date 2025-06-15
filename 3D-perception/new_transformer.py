import torch
import torch.nn as nn

class TransformerProjector(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=768, enc_tokens=2048, dec_tokens=256, out_seq_len=128, num_layers=3):
        super().__init__()
        # 維度投影層
        self.enc_proj = nn.Linear(input_dim, hidden_dim)
        self.dec_proj = nn.Linear(input_dim, hidden_dim)
        
        # 位置編碼
        self.enc_pos_embed = nn.Parameter(torch.randn(enc_tokens, hidden_dim))
        self.dec_pos_embed = nn.Parameter(torch.randn(dec_tokens, hidden_dim))
        
        # [CLS] token 用於全局特徵
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dropout=0.2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 注意力池化
        self.query_embed = nn.Parameter(torch.randn(out_seq_len, hidden_dim))
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=0.2, batch_first=True
        )
        
        # 正規化
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, enc_x, dec_x):  # enc_x: [B, 2048, 256], dec_x: [B, 256, 256]
        B = enc_x.shape[0]
        
        # 投影到 T5 嵌入維度
        enc_x = self.enc_proj(enc_x)  # [B, 2048, 768]
        dec_x = self.dec_proj(dec_x)  # [B, 256, 768]
        
        # 添加位置編碼
        enc_pos = self.enc_pos_embed.unsqueeze(0).expand(B, -1, -1)  # [B, 2048, 768]
        dec_pos = self.dec_pos_embed.unsqueeze(0).expand(B, -1, -1)  # [B, 256, 768]
        enc_x = enc_x + enc_pos  # [B, 2048, 768]
        dec_x = dec_x + dec_pos  # [B, 256, 768]
        
        # 添加 [CLS] token
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, 768]
        
        # 拼接 encoder、decoder 特徵和 [CLS] token
        x = torch.cat([cls_token, enc_x, dec_x], dim=1)  # [B, 1+2048+256=2305, 768]
        
        # Transformer Encoder
        x = self.encoder(x)  # [B, 2305, 768]
        
        # 注意力池化到 out_seq_len
        query = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, out_seq_len, 768]
        x, _ = self.attn_pool(query, x, x)  # [B, out_seq_len, 768]
        x = self.norm(x)  # [B, out_seq_len, 768]
        
        # L2 正規化
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)  # [B, out_seq_len, 768]
        return x