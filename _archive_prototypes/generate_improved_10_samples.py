#!/usr/bin/env python3
"""
IMPROVED: Generate 10+10 samples with retry logic and better prompts
- max_new_tokens: 4096 (increased from 1024)
- temperature: 0.3 (decreased from 0.7 for consistency)
- Retry logic: Auto-retry if <|channel|>final not found
- Multiple in-context examples (1-step, 2-step, 3-step)
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

def extract_final_channel_improved(response: str) -> str:
    """Improved final channel extraction with multiple fallback patterns"""
    # Method 1: Standard <|channel|>final<|message|>...<|return|>
    pattern1 = r'<\|channel\|>final<\|message\>(.*?)<\|return\|>'
    match1 = re.search(pattern1, response, re.DOTALL)
    if match1:
        content = match1.group(1).strip()
        if len(content) > 10:
            return content

    # Method 2: <|channel|>final...<|return|> (without explicit <|message|>)
    pattern2 = r'<\|channel\|>final(.*?)<\|return\|>'
    match2 = re.search(pattern2, response, re.DOTALL)
    if match2:
        content = match2.group(1).strip()
        content = re.sub(r'<\|message\|>', '', content).strip()
        if len(content) > 10:
            return content

    # Method 3: Look for **Step 1:** pattern after "final"
    if 'final' in response.lower():
        parts = response.split('final')
        if len(parts) > 1:
            last_part = parts[-1]
            step_match = re.search(r'\*\*Step 1:\*\*(.*?)(?:<\|return\|>|$)', last_part, re.DOTALL)
            if step_match:
                step_pos = last_part.find('**Step 1:**')
                if step_pos >= 0:
                    content = last_part[step_pos:]
                    content = re.sub(r'<\|return\|>.*$', '', content, flags=re.DOTALL)
                    return content.strip()

    # Method 4: Fallback to last 500 chars if too long
    if len(response) > 500:
        return response[-500:].strip()

    return response.strip()

def is_valid_response(response: str, final_text: str) -> tuple[bool, str]:
    """Check if response is valid, return (is_valid, reason)"""
    if '<|channel|>final' not in response:
        return False, "No <|channel|>final found"

    if '**Step 1:**' not in final_text:
        return False, "No **Step 1:** format found"

    if len(final_text) < 100:
        return False, f"Too short ({len(final_text)} chars)"

    if len(final_text) > 1000:
        return False, f"Too long ({len(final_text)} chars)"

    # Check for grid output (we don't want grids in explanation)
    grid_patterns = [
        r'\d+\s+\d+\s+\d+\s+\d+\s+\d+',  # Grid-like number sequences
        r'Row\s*\d+:',  # Row labels
    ]
    for pattern in grid_patterns:
        if re.search(pattern, final_text):
            return False, "Contains grid output"

    return True, "Valid"

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

def create_type1_prompt_improved(sample_data: dict) -> str:
    """Create IMPROVED TYPE 1 prompt with explicit channel switching examples"""
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

---

**EXAMPLE RESPONSE FORMATS** (showing proper channel usage):

**Example 1 (1-step transformation):**
For (lambda (flipx $0)):

<|channel|>analysis<|message|>
This is horizontal mirroring. Only one step needed.
<|end|><|start|>assistant<|channel|>final<|message|>
**Step 1:** Mirror the entire grid horizontally, swapping left and right sides while preserving all cell colors.

The resulting grid is the horizontal reflection of the original.
<|return|>

**Example 2 (2-step transformation):**
For (lambda (gravity_left $0)):

<|channel|>analysis<|message|>
This moves colored cells left within each row. Two actions needed.
<|end|><|start|>assistant<|channel|>final<|message|>
**Step 1:** Shift each non-zero cell in every row as far left as possible, keeping the original order.
**Step 2:** Leave all other cells as black (0).

All colored cells are now packed to the left of each row.
<|return|>

**Example 3 (3-step transformation):**
For (lambda (ic_center $0)):

<|channel|>analysis<|message|>
This centers the pattern. Need to find bounding box, calculate center, and shift.
<|end|><|start|>assistant<|channel|>final<|message|>
**Step 1:** Identify the smallest rectangle that encloses all non-zero cells in the grid.
**Step 2:** Calculate the center position of the grid dimensions.
**Step 3:** Shift that bounding rectangle to be centered horizontally and vertically.

The resulting grid has the pattern centered with surrounding zeros.
<|return|>

---

**CRITICAL REQUIREMENTS:**
1. ALWAYS use <|channel|>analysis first for brief reasoning (1-2 sentences)
2. Then IMMEDIATELY switch to <|channel|>final<|message|> for your answer
3. In final channel, use **Step 1:**, **Step 2:** format (one sentence per step)
4. Provide ONE final summary sentence
5. Total: 3-4 sentences maximum
6. Do NOT output grids or code in final channel

Your task: Analyze the transformation above and provide your explanation following the EXACT format shown in the examples."""

    return prompt

def create_type2_prompt_improved(sample_data: dict, wrong_program: str, wrong_output: list) -> str:
    """Create IMPROVED TYPE 2 prompt with explicit format"""
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

**INCORRECT Output (from wrong program {wrong_program}):**
{format_grid(wrong_output)}

**CORRECT Output (expected):**
{format_grid(sample_data['test'][0]['output'])}

**Wrong DSL:** {wrong_dsl_desc}
**Correct DSL:** {correct_dsl_desc}

---

**EXAMPLE RESPONSE FORMAT:**

<|channel|>analysis<|message|>
The wrong program did X, but correct program should do Y.
<|end|><|start|>assistant<|channel|>final<|message|>
**Incorrect transformation:** The program applied gravity_right, moving all cells to the far right.

**Step 1:** Identify all cells with the target color value.
**Step 2:** Replace those cells with 0 (black).

The transformation should remove a specific color, not compress the grid.
<|return|>

---

**CRITICAL REQUIREMENTS:**
1. Start with brief analysis in <|channel|>analysis
2. Then switch to <|channel|>final<|message|> for your answer
3. Begin final answer with: **Incorrect transformation:** [one sentence]
4. Then provide correct steps: **Step 1:**, **Step 2:** etc.
5. End with ONE summary sentence
6. Total: 3-4 sentences maximum
7. Do NOT output grids or code

Your task: Explain what went wrong and provide the correct transformation steps."""

    return prompt

def query_gpt_oss_with_retry(prompt: str, model, tokenizer, max_retries: int = 3) -> tuple[str, int, str]:
    """Query GPT-OSS with automatic retry if final channel not found"""

    for attempt in range(max_retries):
        print(f"  Attempt {attempt + 1}/{max_retries}...")

        # Add stronger instruction on retry
        current_prompt = prompt
        if attempt > 0:
            current_prompt += f"\n\n**CRITICAL (RETRY {attempt})**: You MUST use <|channel|>final<|message|> for your answer! Do NOT stay in <|channel|>analysis!"

        # Generate response
        messages = [{"role": "user", "content": current_prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="low"  # Use low reasoning effort for faster generation
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,  # Increased from 1024
            temperature=0.3,      # Decreased from 0.7 for consistency
            top_p=0.9,
            do_sample=True
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Extract final channel
        final_text = extract_final_channel_improved(response)

        # Validate response
        is_valid, reason = is_valid_response(response, final_text)

        if is_valid:
            print(f"  ✓ Valid response on attempt {attempt + 1}")
            return response, attempt + 1, "success"
        else:
            print(f"  ⚠️ Invalid response: {reason}")
            if attempt < max_retries - 1:
                print(f"  Retrying...")

    print(f"  ❌ Failed after {max_retries} attempts")
    return response, max_retries, f"failed: {reason}"

def get_wrong_program(correct_program: str, sample_index: int) -> str:
    """Generate wrong program based on sample index"""
    wrong_primitives_map = {
        'ic_connectX': 'rot270',
        'rot270': 'flipx',
        'gravity_left': 'gravity_right',
        'flipx': 'flipy',
        'mirrorX': 'flipx',
        'ic_erasecol': 'gravity_left',
        'ic_fill': 'ic_center',
    }

    # Samples 8-9: Use multi-step compose (wrong)
    if sample_index >= 7:
        if sample_index == 7:
            return '(lambda (compose flipx gravity_left $0))'
        elif sample_index == 8:
            return '(lambda (compose rot270 ic_fill $0))'
        elif sample_index == 9:
            return '(lambda (compose ic_center flipy gravity_down $0))'

    # Find correct primitive and use wrong one
    for prim in wrong_primitives_map:
        if prim in correct_program:
            wrong_prim = wrong_primitives_map[prim]
            return f'(lambda ({wrong_prim} $0))'

    return '(lambda (flipx $0))'  # fallback

def main():
    print("="*80)
    print("IMPROVED GENERATION: 10+10 SAMPLES WITH RETRY LOGIC")
    print("="*80)
    print("\nImprovements:")
    print("- max_new_tokens: 4096 (was 1024)")
    print("- temperature: 0.3 (was 0.7)")
    print("- Retry logic: Up to 3 attempts")
    print("- Multiple in-context examples (1-step, 2-step, 3-step)")
    print("- Explicit channel switching instructions")
    print("="*80)

    # Load model
    print("\nLoading GPT-OSS model...")
    model_path = "openai/gpt-oss-20b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Let transformers choose device
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
    os.makedirs('/home/ubuntu/TinyRecursiveModels/prototype_10_samples_improved', exist_ok=True)

    type1_results = []
    type2_results = []

    # Track statistics
    type1_attempts = []
    type2_attempts = []

    # Generate for each sample
    for idx, selected in enumerate(selected_samples):
        print(f"\n{'='*80}")
        print(f"SAMPLE {idx+1}/10: {selected['primitive']}")
        print(f"{'='*80}")

        # Get full sample data
        sample_data = helmarc_data[selected['index']]

        # TYPE 1: Correct DSL → Plan
        print(f"\nTYPE 1: Generating plan for correct DSL...")
        prompt1 = create_type1_prompt_improved(sample_data)
        response1_raw, attempts1, status1 = query_gpt_oss_with_retry(prompt1, model, tokenizer, max_retries=3)
        response1 = extract_final_channel_improved(response1_raw)
        type1_attempts.append(attempts1)

        type1_result = {
            'sample_index': idx + 1,
            'task_id': sample_data['task_id'],
            'program': sample_data['program'],
            'primitive': selected['primitive'],
            'prompt': prompt1,
            'raw_response': response1_raw,
            'final_response': response1,
            'attempts': attempts1,
            'status': status1
        }
        type1_results.append(type1_result)

        print(f"✓ TYPE 1 Response ({attempts1} attempts, {status1}):")
        print(f"  {response1[:200]}...")

        # TYPE 2: Wrong DSL → Correction
        print(f"\nTYPE 2: Generating correction for wrong DSL...")
        wrong_program = get_wrong_program(sample_data['program'], idx)
        wrong_output = apply_dsl_program(wrong_program, sample_data['test'][0]['input'])

        if wrong_output:
            prompt2 = create_type2_prompt_improved(sample_data, wrong_program, wrong_output)
            response2_raw, attempts2, status2 = query_gpt_oss_with_retry(prompt2, model, tokenizer, max_retries=3)
            response2 = extract_final_channel_improved(response2_raw)
            type2_attempts.append(attempts2)

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
                'final_response': response2,
                'attempts': attempts2,
                'status': status2
            }
            type2_results.append(type2_result)

            print(f"✓ TYPE 2 Response ({attempts2} attempts, {status2}):")
            print(f"  {response2[:200]}...")
        else:
            print(f"✗ Failed to apply wrong DSL")

    # Save results
    with open('/home/ubuntu/TinyRecursiveModels/prototype_10_samples_improved/type1_correct_dsl_to_plan.json', 'w') as f:
        json.dump(type1_results, f, indent=2)

    with open('/home/ubuntu/TinyRecursiveModels/prototype_10_samples_improved/type2_wrong_dsl_to_correction.json', 'w') as f:
        json.dump(type2_results, f, indent=2)

    # Print statistics
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: prototype_10_samples_improved/")

    print(f"\n=== TYPE 1 Statistics ===")
    type1_success = sum(1 for r in type1_results if r['status'] == 'success')
    print(f"Success rate: {type1_success}/10 ({type1_success*10}%)")
    print(f"Average attempts: {sum(type1_attempts)/len(type1_attempts):.1f}")
    print(f"First-try success: {sum(1 for a in type1_attempts if a == 1)}/10")

    print(f"\n=== TYPE 2 Statistics ===")
    type2_success = sum(1 for r in type2_results if r['status'] == 'success')
    print(f"Success rate: {type2_success}/10 ({type2_success*10}%)")
    print(f"Average attempts: {sum(type2_attempts)/len(type2_attempts):.1f}")
    print(f"First-try success: {sum(1 for a in type2_attempts if a == 1)}/10")

    print(f"\n=== Overall ===")
    total_success = type1_success + type2_success
    print(f"Total success: {total_success}/20 ({total_success*5}%)")
    print(f"Improvement vs previous: {total_success - 14} samples (+{(total_success - 14)*5}%)")

if __name__ == "__main__":
    main()
