import torch
import sys
import types

# # 解決 transformers 的 fsdp import 問題
# fsdp_fake = types.ModuleType("torch.distributed.fsdp")
# sys.modules["torch.distributed.fsdp"] = fsdp_fake

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel
from transformer import TransformerProjector 
from transformers.modeling_outputs import BaseModelOutput
from extract_feature import make_args_parser
from dataset import TestingDataset 
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from detr3d.models import build_model

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

import sng_parser

def extract_object_tokens(caption: str):
    graph = sng_parser.parse(caption)
    obj_words = {ent["head"].lower() for ent in graph["entities"]}
    return obj_words


def object_precision_score(preds, refs):
    """
    Calculate the precision of object tokens in predictions against references,
    and return both mean and standard error.
    """
    precisions = []

    for pred, ref in zip(preds, refs):
        pred_tokens = extract_object_tokens(pred)
        ref_tokens = extract_object_tokens(ref[0])

        if len(ref_tokens) == 0:
            continue  # Skip if no objects in reference

        precision = len(pred_tokens.intersection(ref_tokens)) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        precisions.append(precision)

    if len(precisions) == 0:
        return 0.0, 0.0  # Handle empty case

    mean_precision = np.mean(precisions)
    stderr_precision = np.std(precisions, ddof=1) / np.sqrt(len(precisions))
    return mean_precision, stderr_precision


def object_recall_score(preds, refs):
    """
    Calculate the recall of object tokens in predictions against references,
    and return both mean and standard error.
    """
    recalls = []

    for pred, ref in zip(preds, refs):
        pred_tokens = extract_object_tokens(pred)
        ref_tokens = extract_object_tokens(ref[0])

        if len(ref_tokens) == 0:
            continue  # Skip if no objects in reference

        recall = len(pred_tokens.intersection(ref_tokens)) / len(ref_tokens)
        recalls.append(recall)

    if len(recalls) == 0:
        return 0.0, 0.0

    mean_recall = np.mean(recalls)
    stderr_recall = np.std(recalls, ddof=1) / np.sqrt(len(recalls))
    return mean_recall, stderr_recall


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()

    # Configuration
    dataset_root =  args.dataset_root_dir
    checkpoint_path = args.test_ckpt
    num_points = 20000  # Default number of points to sample from each point cloud
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. load projector
    projector = TransformerProjector(input_dim=256, hidden_dim=768, num_tokens=2304, out_seq_len=128).to(device)
    checkpoint = torch.load("model0_50k_2/epoch_100.pth", map_location=device)
    projector.load_state_dict(checkpoint["model_state_dict"])
    projector.eval()

    # 2. 載入 T5
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    t5 = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    t5.eval()
    # Set pad_token_id and decoder_start_token_id
    t5.config.pad_token_id = tokenizer.pad_token_id
    t5.config.decoder_start_token_id = tokenizer.pad_token_id  # or tokenizer.eos_token_id

    # 3. load pointcloud and extract features
    model_3detr, _ = build_model(args, dataset_config=None)  
    model_3detr.to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu") 
    model_3detr.load_state_dict(checkpoint["model"], strict=False)
    print("finished building model")

    # Load dataset (Extract features on the fly and tokenize captions)
    dataset = TestingDataset(
        root_dir="/project/aimm/ev-honolulu/ds_small",
        feature_extractor=model_3detr, 
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions = []
    references = []
    count = 0

    for batch in dataloader:
        features = batch["3d_feat"]
        # 4. project to text embedding
        with torch.no_grad():
            proj_embed = projector(features)  # [1, seq_len, 768]
            encoder_outputs = BaseModelOutput(last_hidden_state=proj_embed) 
            output_ids = t5.generate(
                encoder_outputs=encoder_outputs,
                max_length=128,
                num_beams=4,        # Optional: improve generation quality
                early_stopping=True
            )
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("Generated Caption:", caption)
        print("Ground Truth Caption:", batch["caption"])
        print("\n")
        predictions.append(caption)
        references.append(batch["caption"])

    print(type(references[0]), references[0])
    # Evaluate the generated captions
    bleu_score = corpus_bleu(references, predictions)
    bleu_stderr = np.std([corpus_bleu([r], [p]) for r, p in zip(references, predictions)]) / np.sqrt(len(predictions))
    print(f"BLEU Score: {bleu_score:.4f} ± {bleu_stderr:.4f}")
    meteor_scores = [
        meteor_score([word_tokenize(ref[0])], word_tokenize(pred))
        for ref, pred in zip(references, predictions)
    ]
    avg_meteor = np.mean(meteor_scores)
    stderr_meteor = np.std(meteor_scores, ddof=1) / np.sqrt(len(meteor_scores))
    print(f"METEOR Score: {avg_meteor:.4f} ± {stderr_meteor:.4f}")
    rouge = Rouge()
    scores = rouge.get_scores(predictions, [r[0] for r in references], avg=False)
    rouge1_scores = [s['rouge-1']['f'] for s in scores]
    rouge2_scores = [s['rouge-2']['f'] for s in scores]
    rougeL_scores = [s['rouge-l']['f'] for s in scores]

    print(f"ROUGE-1: {np.mean(rouge1_scores):.4f} ± {np.std(rouge1_scores, ddof=1) / np.sqrt(len(rouge1_scores)):.4f}")
    print(f"ROUGE-2: {np.mean(rouge2_scores):.4f} ± {np.std(rouge2_scores, ddof=1) / np.sqrt(len(rouge2_scores)):.4f}")
    print(f"ROUGE-L: {np.mean(rougeL_scores):.4f} ± {np.std(rougeL_scores, ddof=1) / np.sqrt(len(rougeL_scores)):.4f}")

    object_precision_mean, object_precision_stderr = object_precision_score(predictions, references)
    print(f"Object Precision Score: {object_precision_mean:.4f} ± {object_precision_stderr:.4f}")

    object_recall_mean, object_recall_stderr = object_recall_score(predictions, references)
    print(f"Object Recall Score: {object_recall_mean:.4f} ± {object_recall_stderr:.4f}")
