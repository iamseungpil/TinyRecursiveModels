#!/usr/bin/env python3
"""
Generate 10+10 samples for two dataset types:
TYPE 1: Correct DSL → Step-by-step plan
TYPE 2: Wrong DSL → Corrected step-by-step plan
"""

import json
import sys
import torch
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add paths
sys.path.insert(0, '/home/ubuntu/dreamcoder-arc/ec')
from dreamcoder.domains.arc.arcPrimitivesIC2 import *

# DSL Primitive descriptions
DSL_DESCRIPTIONS = {
    "ic_connectX": "Fills the horizontal gap between two separate blocks of the same color",
    "rot270": "Rotates the grid 270 degrees clockwise (90 degrees counter-clockwise)",
    "gravity_left": "Moves all colored cells to the left within each row, preserving their order",
    "flipx": "Mirrors the grid horizontally, swapping left and right sides",
    "mirrorX": "Creates a horizontal mirror reflection of the grid",
    "ic_erasecol": "Replaces all cells of a specific color with black (0)",
    "ic_fill": "Fills empty cells within colored regions",
    "gravity_down": "Moves all colored cells downward within each column, preserving their order",
    "flipy": "Mirrors the grid vertically, swapping top and bottom sides",
    "ic_center": "Centers the colored pattern within the grid",
    "gravity_right": "Moves all colored cells to the right within each row",
    "gravity_up": "Moves all colored cells upward within each column",
    "ic_compress2": "Removes empty rows and columns while preserving relative positions",
}

def extract_final_channel(response: str) -> str:
    """Extract only the final channel response from GPT-OSS output"""
    # Pattern 1: <|channel|>final<|message|>...<|return|>
    pattern1 = r'<\|channel\|>final<\|message\>(.*?)<\|return\|>'
    match = re.search(pattern1, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Pattern 2: assistantfinalTEXT (when markers appear as text)
    pattern2 = r'assistantfinal(.*?)(?:<\|return\|>|$)'
    match2 = re.search(pattern2, response, re.DOTALL)
    if match2:
        return match2.group(1).strip()

    # Pattern 3: Look for final after analysis
    pattern3 = r'analysis.*?final(.*?)(?:<\|return\|>|$)'
    match3 = re.search(pattern3, response, re.DOTALL | re.IGNORECASE)
    if match3:
        return match3.group(1).strip()

    # Fallback: return last 500 chars if response is too long
    if len(response) > 500:
        return response[-500:]
    return response

def format_grid(grid):
    """Format grid as string"""
    return "\n".join([" ".join([str(cell) for cell in row]) for row in grid])

def get_dsl_description(program: str) -> str:
    """Get natural language description of DSL program"""
    for prim, desc in DSL_DESCRIPTIONS.items():
        if prim in program:
            return f"{prim}: {desc}"
    return ""

def apply_dsl_program(program_str: str, input_grid: list) -> list:
    """Apply DSL program to input grid"""
    try:
        input_array = np.array(input_grid)
        grid_obj = Grid(input_array)

        # Parse program and apply
        if 'rot270' in program_str:
            result = rot270(grid_obj)
        elif 'ic_connectX' in program_str:
            result = ic_connectX(grid_obj)
        elif 'gravity_left' in program_str:
            result = gravity_left(grid_obj)
        elif 'gravity_right' in program_str:
            result = gravity_right(grid_obj)
        elif 'gravity_down' in program_str:
            result = gravity_down(grid_obj)
        elif 'gravity_up' in program_str:
            result = gravity_up(grid_obj)
        elif 'flipx' in program_str:
            result = flipx(grid_obj)
        elif 'flipy' in program_str:
            result = flipy(grid_obj)
        elif 'mirrorX' in program_str:
            result = mirrorX(grid_obj)
        elif 'ic_erasecol' in program_str:
            # Extract color parameter
            if 'c1' in program_str:
                result = ic_erasecol(c1, grid_obj)
            elif 'c2' in program_str:
                result = ic_erasecol(c2, grid_obj)
            else:
                result = ic_erasecol(c1, grid_obj)
        elif 'ic_fill' in program_str:
            result = ic_fill(grid_obj)
        elif 'ic_center' in program_str:
            result = ic_center(grid_obj)
        else:
            return None

        return result.grid.tolist()
    except Exception as e:
        print(f"Error applying DSL: {e}")
        return None

def create_type1_prompt(sample_data: dict) -> str:
    """Create TYPE 1 prompt (Correct DSL → Plan)"""
    dsl_desc = get_dsl_description(sample_data['program'])

    prompt = f"""You are analyzing an ARC (Abstract Reasoning Corpus) puzzle transformation.

**Training Examples:**

Example 1 Input:
{format_grid(sample_data['examples'][0]['input'])}

Example 1 Output:
{format_grid(sample_data['examples'][0]['output'])}

Example 2 Input:
{format_grid(sample_data['examples'][1]['input'])}

Example 2 Output:
{format_grid(sample_data['examples'][1]['output'])}

**Test Input:**
{format_grid(sample_data['test'][0]['input'])}

**Transformation Program:** {sample_data['program']}
**DSL Information:** {dsl_desc}

**Example Response Format:**

For a horizontal mirroring task (lambda (flipx $0)):

**Step 1:** Mirror the entire grid horizontally, swapping left and right sides while preserving all cell colors.

This is a single-step geometric transformation that creates a horizontal reflection.

---

Your task: Provide a CONCISE explanation (3-4 sentences maximum) that includes:
1. Each transformation step in **Step N:** format
2. A final summary sentence

Requirements:
- Use **Step 1:**, **Step 2:**, etc. format for each step
- Keep each step description to ONE sentence
- End with ONE summary sentence
- Do NOT output grids or code
- Maximum 4 sentences total"""

    return prompt

def create_type2_prompt(sample_data: dict, wrong_program: str, wrong_output: list) -> str:
    """Create TYPE 2 prompt (Wrong DSL → Correction)"""
    correct_dsl_desc = get_dsl_description(sample_data['program'])
    wrong_dsl_desc = get_dsl_description(wrong_program)

    prompt = f"""You are analyzing an ARC (Abstract Reasoning Corpus) puzzle transformation.

**Training Examples (showing CORRECT transformation):**

Example 1 Input:
{format_grid(sample_data['examples'][0]['input'])}

Example 1 Output:
{format_grid(sample_data['examples'][0]['output'])}

Example 2 Input:
{format_grid(sample_data['examples'][1]['input'])}

Example 2 Output:
{format_grid(sample_data['examples'][1]['output'])}

**Test Input:**
{format_grid(sample_data['test'][0]['input'])}

**INCORRECT Output (from wrong transformation {wrong_program}):**
{format_grid(wrong_output)}

**CORRECT Output (expected):**
{format_grid(sample_data['test'][0]['output'])}

**Wrong DSL Information:** {wrong_dsl_desc}
**Correct DSL Information:** {correct_dsl_desc}

**Example Response Format:**

**Incorrect transformation:** gravity_left only compresses cells leftward within rows.
**Step 1:** Identify all cells with color value 1 and replace them with 0 (black).

The transformation should remove a specific color, not compress the grid.

---

Your task: Provide a CONCISE explanation (3-4 sentences maximum) that includes:
1. What the incorrect transformation did (one sentence)
2. The correct steps in **Step N:** format
3. A final summary of the difference

Requirements:
- Start with **Incorrect transformation:** description
- Use **Step 1:**, **Step 2:**, etc. for correct steps
- End with ONE summary sentence
- Do NOT output grids or code
- Maximum 4 sentences total"""

    return prompt

def query_gpt_oss(prompt: str, model, tokenizer) -> str:
    """Query GPT-OSS model"""
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return response

def get_wrong_program(correct_program: str, sample_index: int) -> str:
    """Generate wrong program based on sample index"""
    # Samples 1-7: Use different 1-step primitive
    wrong_primitives_map = {
        'ic_connectX': 'rot270',
        'rot270': 'flipx',
        'gravity_left': 'gravity_right',
        'flipx': 'flipy',
        'mirrorX': 'flipx',
        'ic_erasecol': 'gravity_left',
        'ic_fill': 'ic_center',
    }

    # Samples 8-9: Use 2-step compose (wrong)
    if sample_index >= 7:
        if sample_index == 7:
            return '(lambda (compose flipx gravity_left $0))'
        elif sample_index == 8:
            return '(lambda (compose rot270 ic_fill $0))'
        elif sample_index == 9:
            return '(lambda (compose ic_center flipy gravity_down $0))'

    # Find correct primitive
    for prim in wrong_primitives_map:
        if prim in correct_program:
            wrong_prim = wrong_primitives_map[prim]
            if 'c1' in correct_program:
                return f'(lambda ({wrong_prim} $0))'
            return f'(lambda ({wrong_prim} $0))'

    return '(lambda (flipx $0))'  # fallback

def main():
    print("="*80)
    print("GENERATING 10+10 SAMPLES FOR TWO DATASET TYPES")
    print("="*80)

    # Load model
    print("\nLoading GPT-OSS model...")
    model_path = "openai/gpt-oss-20b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    print("✓ Model loaded")

    # Load selected samples
    with open('/home/ubuntu/TinyRecursiveModels/selected_10_samples.json', 'r') as f:
        selected_samples = json.load(f)

    # Load full HelmARC data
    with open('/data/helmarc_correct/20251024_062500/samples.json', 'r') as f:
        helmarc_data = json.load(f)

    # Create output directory
    import os
    os.makedirs('/home/ubuntu/TinyRecursiveModels/prototype_10_samples', exist_ok=True)

    type1_results = []
    type2_results = []

    # Generate for each sample
    for idx, selected in enumerate(selected_samples):
        print(f"\n{'='*80}")
        print(f"SAMPLE {idx+1}/10: {selected['primitive']}")
        print(f"{'='*80}")

        # Get full sample data
        sample_data = helmarc_data[selected['index']]

        # TYPE 1: Correct DSL → Plan
        print(f"\nTYPE 1: Generating plan for correct DSL...")
        prompt1 = create_type1_prompt(sample_data)
        response1_raw = query_gpt_oss(prompt1, model, tokenizer)
        response1 = extract_final_channel(response1_raw)

        type1_result = {
            'sample_index': idx + 1,
            'task_id': sample_data['task_id'],
            'program': sample_data['program'],
            'primitive': selected['primitive'],
            'prompt': prompt1,
            'raw_response': response1_raw,
            'final_response': response1
        }
        type1_results.append(type1_result)

        print(f"✓ TYPE 1 Response:")
        print(f"  {response1[:200]}...")

        # TYPE 2: Wrong DSL → Correction
        print(f"\nTYPE 2: Generating correction for wrong DSL...")
        wrong_program = get_wrong_program(sample_data['program'], idx)
        wrong_output = apply_dsl_program(wrong_program, sample_data['test'][0]['input'])

        if wrong_output:
            prompt2 = create_type2_prompt(sample_data, wrong_program, wrong_output)
            response2_raw = query_gpt_oss(prompt2, model, tokenizer)
            response2 = extract_final_channel(response2_raw)

            type2_result = {
                'sample_index': idx + 1,
                'task_id': sample_data['task_id'],
                'correct_program': sample_data['program'],
                'wrong_program': wrong_program,
                'primitive': selected['primitive'],
                'test_input': sample_data['test'][0]['input'],
                'wrong_output': wrong_output,
                'correct_output': sample_data['test'][0]['output'],
                'prompt': prompt2,
                'raw_response': response2_raw,
                'final_response': response2
            }
            type2_results.append(type2_result)

            print(f"✓ TYPE 2 Response:")
            print(f"  {response2[:200]}...")
        else:
            print(f"✗ Failed to apply wrong DSL")

    # Save results
    with open('/home/ubuntu/TinyRecursiveModels/prototype_10_samples/type1_correct_dsl_to_plan.json', 'w') as f:
        json.dump(type1_results, f, indent=2)

    with open('/home/ubuntu/TinyRecursiveModels/prototype_10_samples/type2_wrong_dsl_to_correction.json', 'w') as f:
        json.dump(type2_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"DONE! Generated {len(type1_results)} TYPE 1 and {len(type2_results)} TYPE 2 samples")
    print(f"{'='*80}")
    print(f"\nResults saved to:")
    print(f"  - prototype_10_samples/type1_correct_dsl_to_plan.json")
    print(f"  - prototype_10_samples/type2_wrong_dsl_to_correction.json")

if __name__ == "__main__":
    main()
