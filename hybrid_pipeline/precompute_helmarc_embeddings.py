"""
Pre-compute GPT-OSS embeddings for HelmARC analysis.

Usage:
    python precompute_helmarc_embeddings.py \
        --analysis_dir /data/helmarc/analysis \
        --identifiers_path /data/helmarc_trm/identifiers.json \
        --output_path /data/helmarc_gptoss_embeddings.pt \
        --model_name unsloth/gpt-oss-mxfp4-20b
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def precompute_embeddings(
    analysis_dir: str,
    output_path: str,
    identifiers_path: str,
    model_name: str = "unsloth/gpt-oss-mxfp4-20b",
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    use_fast_tokenizer: bool = False,
):
    """
    Pre-compute LLaMA embeddings for all HelmARC GPT-OSS analysis texts.

    Args:
        analysis_dir: Directory containing analysis_XXXX.json files
        output_path: Where to save embeddings (*.pt file)
        identifiers_path: Path to identifiers.json for task_id → puzzle_identifier mapping
        model_name: LLaMA model identifier
        device: Device to use
    """
    # Load identifiers mapping
    print(f"📥 Loading identifiers from {identifiers_path}")
    with open(identifiers_path) as f:
        identifiers_list = json.load(f)

    # Create reverse mapping: task_id → puzzle_identifier
    task_to_id = {}
    for idx, task_id in enumerate(identifiers_list):
        if task_id != "<blank>":
            # Remove augmentation suffix if present (e.g., "task|||t1|||..." → "task")
            base_task_id = task_id.split("|||")[0] if "|||" in task_id else task_id
            task_to_id[base_task_id] = idx

    print(f"✅ Loaded {len(task_to_id)} task ID mappings")

    print(f"📦 Loading model: {model_name}")
    dtype = torch_dtype
    if isinstance(dtype, str) and dtype.lower() != "auto":
        dtype = getattr(torch, dtype)
    elif isinstance(dtype, str) and dtype.lower() == "auto":
        dtype = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast_tokenizer,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True
    )
    if device_map != "auto":
        model.to(device)
    model.eval()
    print(f"✅ Model ready ({device})")

    embeddings = {}
    analysis_files = sorted(Path(analysis_dir).glob("analysis_*.json"))

    print(f"📊 Processing {len(analysis_files)} analysis files...")

    for idx, analysis_file in enumerate(tqdm(analysis_files)):
        try:
            with open(analysis_file) as f:
                data = json.load(f)

            # Extract task_id from metadata
            task_id = data["sample_metadata"]["task_id"]

            # Get puzzle_identifier for this task
            if task_id not in task_to_id:
                print(f"⚠️  Warning: {task_id} not found in identifiers, skipping")
                continue

            puzzle_identifier = task_to_id[task_id]

            # Get text (use raw_response as fallback if full_response is missing)
            text = data.get("full_response") or data.get("raw_response") or ""

            if not text:
                print(f"⚠️  Warning: {task_id} has no text, skipping")
                continue

            # Encode with LLaMA
            with torch.no_grad():
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(device)

                outputs = model(**inputs, output_hidden_states=True)
                z_llm = outputs.hidden_states[-1][:, -1, :].cpu()  # [1, 4096]

            embeddings[puzzle_identifier] = z_llm

            # Periodic cleanup and checkpoint
            if (idx + 1) % 100 == 0:
                torch.cuda.empty_cache()
                # Save intermediate checkpoint
                checkpoint_path = output_path.replace(".pt", f"_checkpoint_{idx+1}.pt")
                torch.save(embeddings, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")

        except Exception as e:
            print(f"❌ Error processing {analysis_file.name}: {e}")
            continue

    # Save
    print(f"💾 Saving embeddings to {output_path}")
    torch.save(embeddings, output_path)
    print(f"✅ Saved {len(embeddings)} embeddings!")

    # Print stats
    sample_embedding = list(embeddings.values())[0]
    print(f"\n📊 Embedding shape: {sample_embedding.shape}")
    print(f"📊 Total size: {Path(output_path).stat().st_size / 1e9:.2f} GB")


def str2bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "t", "yes", "y"}


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute GPT-OSS embeddings for HelmARC"
    )
    parser.add_argument("--analysis_dir", type=str, default="/data/helmarc/analysis")
    parser.add_argument("--output_path", type=str, default="/data/helmarc_embeddings.pt")
    parser.add_argument("--identifiers_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="unsloth/gpt-oss-mxfp4-20b")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", type=str, default="True")
    parser.add_argument("--use_fast_tokenizer", type=str, default="False")

    args = parser.parse_args()

    precompute_embeddings(
        analysis_dir=args.analysis_dir,
        output_path=args.output_path,
        identifiers_path=args.identifiers_path,
        model_name=args.model_name,
        device=args.device,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=str2bool(args.trust_remote_code),
        use_fast_tokenizer=str2bool(args.use_fast_tokenizer),
    )


if __name__ == "__main__":
    main()
