#!/usr/bin/env python3
"""
Batch analyze HelmARC samples with GPT-OSS
Processes a range of samples on a specific GPU
"""

import argparse
import json
import os
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch

# Import the analyzer
from gpt_oss_analyzer import GPTOSSARCAnalyzer

def setup_logging(output_dir: str, gpu: int):
    """Setup logging to both file and console"""
    log_file = os.path.join(output_dir, f'analysis_gpu{gpu}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Batch analyze HelmARC samples with GPT-OSS')
    parser.add_argument('--input', type=str, required=True, help='Input HelmARC samples JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for analysis results')
    parser.add_argument('--gpu', type=int, required=True, help='GPU device ID')
    parser.add_argument('--start', type=int, required=True, help='Start sample index (inclusive)')
    parser.add_argument('--end', type=int, required=True, help='End sample index (exclusive)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Setup logging
    logger = setup_logging(args.output, args.gpu)

    logger.info("=" * 60)
    logger.info("GPT-OSS Post-Processing for Helmholtz Samples")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Range: {args.start} - {args.end}")
    logger.info("=" * 60)

    # Load samples
    logger.info(f"Loading Helmholtz samples from {args.input}...")
    with open(args.input, 'r') as f:
        all_samples = json.load(f)

    logger.info(f"Loaded {len(all_samples)} samples")

    # Get subset for this GPU
    samples = all_samples[args.start:args.end]
    logger.info(f"Processing samples {args.start} to {args.end} ({len(samples)} samples)")

    # Initialize analyzer on specific GPU
    logger.info(f"Initializing GPT-OSS analyzer on cuda:{args.gpu}...")
    analyzer = GPTOSSARCAnalyzer(
        model_path="openai/gpt-oss-20b",
        device=f"cuda:{args.gpu}"
    )

    # Process samples
    logger.info("Starting analysis...")
    results = []

    for i, sample in enumerate(tqdm(samples, desc="Analyzing")):
        sample_idx = args.start + i
        task_id = sample.get('task_id', f'sample_{sample_idx}')
        program = sample.get('program', 'unknown')

        try:
            # Get expected output
            test_output = sample['test'][0]['output']
            import numpy as np
            expected_output = f"Test Output 1:\n{analyzer.grid_to_string(np.array(test_output))}"

            # Analyze (with max_retries=5)
            result = analyzer.analyze_arc_data(
                helmholtz_data=sample,
                program=program,
                expected_output=expected_output,
                max_retries=5
            )

            # Skip if analysis failed
            if result is None:
                logger.warning(f"⚠️  Skipping sample {task_id} - analysis failed after max retries")
                continue

            # Add metadata
            result['task_id'] = task_id
            result['program'] = program
            result['sample_index'] = sample_idx
            result['timestamp'] = datetime.now().isoformat()

            results.append(result)

            # Save incrementally every 10 samples
            if (i + 1) % 10 == 0:
                output_file = os.path.join(args.output, f'analysis_checkpoint_{sample_idx}.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"Checkpoint saved: {len(results)} samples analyzed")

        except Exception as e:
            logger.error(f"Failed to analyze sample {task_id}: {e}")
            continue

    # Save final results
    output_file = os.path.join(args.output, f'analysis_final_gpu{args.gpu}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Analysis complete! Processed {len(results)}/{len(samples)} samples")
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
