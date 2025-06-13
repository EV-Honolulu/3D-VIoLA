import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from scene_dataset import SceneCaptionDataset
from detr3d.models import build_model
from transformers import T5Tokenizer, T5EncoderModel
from transformers.modeling_outputs import BaseModelOutput
from dataset import TraningDataset
from transformer import TransformerProjector
from tqdm import tqdm
from transformers import T5ForConditionalGeneration
import torch.nn.functional as F
from extract_feature import make_args_parser
import wandb
import sng_parser

def extract_object_tokens(caption: str, tokenizer):
    graph = sng_parser.parse(caption)
    obj_words = {ent["head"].lower() for ent in graph["entities"]}
    return obj_words

def get_object_token_ids(obj_words, tokenizer):
    # e.g., "cup" → token ID 3025
    obj_token_ids = set()
    for word in obj_words:
        token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        obj_token_ids.update(token_ids)
    return obj_token_ids

def soft_object_alignment_loss(proj_out, input_ids, refs, tokenizer):
    B, L, D = proj_out.shape
    losses = []

    for i in range(B):
        ref_text = refs[i]
        obj_words = extract_object_tokens(ref_text, tokenizer)
        obj_token_ids = get_object_token_ids(obj_words, tokenizer)

        # Skip if no objects
        if len(obj_token_ids) == 0:
            continue

        mask = torch.tensor([
            1.0 if tok.item() in obj_token_ids else 0.0 
            for tok in input_ids[i]
        ], device=proj_out.device)

        if mask.sum() == 0:
            continue  # no match

        proj_i = proj_out[i]                    # [L, D]
        proj_i = F.normalize(proj_i, dim=-1)    # cosine sim
        obj_emb = proj_i[mask.bool()]           # [#obj, D]

        avg_sim = (obj_emb @ proj_i.T).mean()   # [#obj, L] → mean sim to all
        losses.append(1 - avg_sim)              # want high similarity

    if len(losses) == 0:
        return torch.tensor(0.0, device=proj_out.device)
    return torch.stack(losses).mean()



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

def semantic_alignment_loss(proj_out, encoder_hidden):
    proj_pooled = proj_out.mean(dim=1)
    text_pooled = encoder_hidden.mean(dim=1)

    proj_norm = F.normalize(proj_pooled, dim=-1)
    text_norm = F.normalize(text_pooled, dim=-1)

    cos_sim = (proj_norm * text_norm).sum(dim=-1)  # [B]
    loss = 1 - cos_sim.mean()
    return loss

def tokenwise_semantic_loss(proj_out, encoder_hidden):
    proj_norm = F.normalize(proj_out, dim=-1)     # [B, L, 768]
    text_norm = F.normalize(encoder_hidden, dim=-1)

    cos_sim = (proj_norm * text_norm).sum(dim=-1)  # [B, L]
    return 1 - cos_sim.mean()

if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()

    # Configuration
    dataset_root = ["/project/aimm/ev-honolulu/dataset/train/pick_and_place_simple/ds_large/", "/project/aimm/ev-honolulu/dataset/train/look_at_obj_in_light/ds_large/", "/project/aimm/ev-honolulu/dataset/new/ds_large/"]
    checkpoint_path = args.test_ckpt
    batch_size = 8#8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model_3detr, _ = build_model(args, dataset_config=None)  # Use the same build_model function as in main.py
    model_3detr.to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu") 
    model_3detr.load_state_dict(checkpoint["model"], strict=False)
    
    for param in model_3detr.parameters():
        param.requires_grad = False

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    t5 = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    t5.eval()
    for param in t5.parameters():
        param.requires_grad = False

    t5_encoder = t5.encoder
    t5_encoder.eval()
    for param in t5_encoder.parameters():
        param.requires_grad = False

    # train the decoder
    # for name, param in t5.named_parameters():
    #     if name.startswith("decoder.") or name.startswith("lm_head"):
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    # t5.decoder.train()
    # t5.lm_head.train()

    dataset = TraningDataset(
        num_points=50000,  # 20000
        root_dirs = dataset_root,
        tokenizer=tokenizer, 
        feature_extractor=model_3detr, 
    )
    print("len", len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    output_dir = "model0_50k_2"
    os.makedirs(output_dir, exist_ok=True)
    
    projector = TransformerProjector(input_dim=256, hidden_dim=768, num_tokens=2304, out_seq_len=128).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-4)
    checkpoint_path = "model0_50k_1/epoch_100.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        projector.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded existing projector from {checkpoint_path}")

    # decoder_params = [p for n, p in t5.named_parameters() if p.requires_grad]
    # optimizer = torch.optim.AdamW(
    #     list(projector.parameters()) + decoder_params, lr=1e-4
    # )

    projector.train()  
    epochs = 300 
    wandb.init(project="3detr_captioning", name="projector_training")
    for epoch in range(epochs):
        total_loss = 0
        loss1 = 0
        loss2 = 0
        loss3 = 0
        epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch in epoch_bar:
            x = batch["3d_feat"].to(device)                # [B, N, 256]
            
            input_ids = batch["input_ids"].to(device)      # [B, L]
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                encoder_hidden = t5_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state  # [B, L, 768]

            proj_out = projector(x)  # [B, L, 768]

            # Loss 1: hidden state alignment (MSE)
            loss_embed = F.mse_loss(proj_out, encoder_hidden)

            # Loss 2: decoder generation loss
            outputs = t5(
                encoder_outputs=(proj_out,),
                labels=input_ids,
                decoder_attention_mask=attention_mask,
            )
            loss_gen = outputs.loss

            # Loss 3: semantic alignment loss
            loss_cos = tokenwise_semantic_loss(proj_out, encoder_hidden)

            # Loss 4: scene graph alignment loss
            preds = t5.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=proj_out),
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
            preds = [tokenizer.decode(p, skip_special_tokens=True) for p in preds]
            refs = [tokenizer.decode(r, skip_special_tokens=True) for r in input_ids]
            # print("preds", preds)
            # print("refs", refs)
            loss_sg = soft_object_alignment_loss(proj_out, input_ids, refs, tokenizer)
            
            # if epoch < 10:
            #     loss = loss_cos
            # elif epoch < 40:
            #     loss = loss_cos + 1.0 * loss_gen
            # else:
            #     loss = loss_cos + 1.0 * loss_gen + 1.0 * loss_embed

            loss = loss_gen * 0.5 + loss_sg * 0.5

            wandb.log({
                "loss": loss.item(),
                # "loss_embed": loss_embed.item(),
                "loss_gen": loss_gen.item(),
                # "loss_cos": loss_cos.item(),
                "loss_sg": loss_sg.item(),
            })
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss1 += loss_embed.item()
            loss2 += loss_gen.item()
            loss3 += loss_cos.item()
            # Update tqdm with current loss
            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")


        print(f"Epoch {epoch+1} average loss: {total_loss / len(dataloader):.4f}, loss_embed: {loss1 / len(dataloader):.4f}, loss_gen: {loss2 / len(dataloader):.4f}, loss_cos: {loss3 / len(dataloader):.4f}")
        if (epoch + 1) % 5 == 0:
            checkpoint_filename = os.path.join(output_dir, f"epoch_{epoch+1}.pth")
            torch.save({
                "model_state_dict": projector.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_filename)
            print(f"Checkpoint saved: {checkpoint_filename}")
