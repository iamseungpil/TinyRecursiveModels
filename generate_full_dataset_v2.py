#!/usr/bin/env python3
"""
Generate full dataset V2 with CORRECTED DSL descriptions
- Complete 67 DSL primitive descriptions (based on DSL_PRIMITIVES_REFERENCE.md)
- CORRECTED flipx/flipy definitions
- Enhanced validation logic
- Improved in-context examples
"""

import json
import sys
import torch
import numpy as np
import re
import argparse
import os
import gc
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add paths
sys.path.insert(0, '/home/ubuntu/dreamcoder-arc/ec')
from dreamcoder.domains.arc.arcPrimitivesIC2 import *

# COMPLETE DSL Primitive descriptions (67 primitives)
# Based on DSL_PRIMITIVES_REFERENCE.md with CORRECTED flipx/flipy
COMPLETE_DSL_DESCRIPTIONS = {
    # Rotation primitives
    "rot90": "Rotates the grid 90 degrees clockwise",
    "rot180": "Rotates the grid 180 degrees",
    "rot270": "Rotates the grid 270 degrees clockwise (or 90 degrees counter-clockwise)",
    "swapxy": "Transposes the grid, swapping rows and columns (diagonal flip)",

    # Flip primitives (CORRECTED - CRITICAL!)
    "flipx": "Flips the grid vertically (upside-down), swapping top and bottom rows",
    "flipy": "Flips the grid horizontally (left-right), swapping left and right columns",

    # Mirror primitives (expand by mirroring)
    "mirror": "Creates a mirrored copy of the grid",
    "mirrorX": "Extends the grid by mirroring horizontally (creates left-right symmetry)",
    "mirrorY": "Extends the grid by mirroring vertically (creates top-bottom symmetry)",

    # Gravity primitives
    "gravity_left": "Moves all colored cells to the left within each row, like gravity pulling leftward",
    "gravity_right": "Moves all colored cells to the right within each row, like gravity pulling rightward",
    "gravity_up": "Moves all colored cells upward within each column, like gravity pulling upward",
    "gravity_down": "Moves all colored cells downward within each column, like gravity pulling downward",

    # Half selection primitives
    "top_half": "Returns the top half of the grid",
    "bottom_half": "Returns the bottom half of the grid",
    "left_half": "Returns the left half of the grid",
    "right_half": "Returns the right half of the grid",

    # Color operations
    "colour": "Returns or identifies color values",
    "count": "Counts occurrences of a pattern or color",
    "get_bg": "Identifies and returns cells matching the background color",
    "rarestcol": "Finds the rarest (least frequent) color in the grid",
    "set_bg": "Sets the background color to a specified value",
    "setcol": "Sets all cells to a specific color",
    "topcol": "Identifies the color of the topmost non-zero cell",

    # IC (Image Correction/Connected Components) primitives
    "ic_fill": "Fills interior empty cells within colored regions",
    "ic_erasecol": "Removes all cells of a specific color (replaces with black/0)",
    "ic_center": "Centers the colored pattern within the grid",
    "ic_compress2": "Removes empty rows and columns to compress the grid (level 2)",
    "ic_compress3": "Removes empty rows and columns to compress the grid (level 3)",
    "ic_embed": "Embeds one grid pattern into another",
    "ic_interior": "Extracts only the interior cells of colored regions",
    "ic_invert": "Inverts the color pattern",
    "ic_makeborder": "Creates a border around colored regions",
    "ic_connect": "Connects separated components of the same color",
    "ic_composegrowing": "Composes patterns with growing size",
    "ic_toorigin": "Moves all colored cells to the origin (top-left)",
    "ic_filtercol": "Filters cells by color, keeping only specified colors",
    "ic_pickunique": "Selects unique color patterns from the grid",
    "ic_connectX": "Fills the horizontal gap between two separate blocks of the same color",
    "ic_connectY": "Fills the vertical gap between two separate blocks of the same color",

    # IC split primitives
    "ic_splitall": "Splits the grid into all connected components",
    "ic_splitrows": "Splits the grid by rows into separate subgrids",
    "ic_splitcolumns": "Splits the grid by columns into separate subgrids",
    "ic_splitcols": "Splits by color groups",

    # Pick/select primitives
    "pickmax_size": "Selects the largest component by size",
    "pickmax_count": "Selects the component with maximum count",
    "pickmax_cols": "Selects columns with maximum values",
    "pickmax_interior_count": "Selects component with maximum interior cell count",
    "pickmax_neg_size": "Selects component with maximum negative size metric",
    "pickmax_neg_count": "Selects component with maximum negative count",
    "pickmax_neg_interior_count": "Selects component with maximum negative interior count",
    "pickmax_x_pos": "Selects component with maximum x position",
    "pickmax_x_neg": "Selects component with minimum x position",
    "pickmax_y_pos": "Selects component with maximum y position",
    "pickmax_y_neg": "Selects component with minimum y position",
    "pickcommon": "Selects the most common pattern from multiple options",

    # Repeat primitives
    "repeat": "Repeats the pattern multiple times",
    "repeatX": "Repeats the grid horizontally",
    "repeatY": "Repeats the grid vertically",

    # Higher-order functions
    "map": "Applies a function to each element",
    "mapSplit8": "Maps a function over 8 split parts",
    "lcons": "List constructor operation",
    "mklist": "Creates a list from elements",

    # Overlay
    "overlay": "Overlays one grid pattern on top of another",
    "fillobj": "Fills objects with a specific pattern",

    # Logical operations
    "logical_and": "Performs logical AND operation between two grids",

    # Utility
    "split8": "Splits the grid into 8 parts",
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

def validate_flipx_flipy_description(program: str, final_text: str) -> tuple[bool, str]:
    """
    CRITICAL: Validate flipx/flipy descriptions are correct

    flipx should mention: "vertically", "upside-down", or "top and bottom"
    flipx should NOT mention: "horizontally" or "left and right"

    flipy should mention: "horizontally" or "left-right" or "left and right"
    flipy should NOT mention: "vertically" or "top and bottom"
    """
    final_lower = final_text.lower()

    # Check flipx
    if 'flipx' in program:
        # Should contain vertical indicators
        has_vertical = any(word in final_lower for word in ['vertically', 'upside-down', 'upside down', 'top and bottom'])
        # Should NOT contain horizontal indicators
        has_horizontal = any(word in final_lower for word in ['horizontally', 'left and right', 'left-right'])

        if has_horizontal and not has_vertical:
            return False, "flipx incorrectly described as horizontal (should be vertical)"
        if not has_vertical:
            return False, "flipx missing vertical flip description"

    # Check flipy
    if 'flipy' in program:
        # Should contain horizontal indicators
        has_horizontal = any(word in final_lower for word in ['horizontally', 'left and right', 'left-right'])
        # Should NOT contain vertical indicators
        has_vertical = any(word in final_lower for word in ['vertically', 'upside-down', 'upside down', 'top and bottom'])

        if has_vertical and not has_horizontal:
            return False, "flipy incorrectly described as vertical (should be horizontal)"
        if not has_horizontal:
            return False, "flipy missing horizontal flip description"

    return True, "Valid"

def is_valid_response(response: str, final_text: str, program: str = "") -> tuple[bool, str]:
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

    # CRITICAL: Validate flipx/flipy descriptions
    if program:
        is_valid_flip, flip_reason = validate_flipx_flipy_description(program, final_text)
        if not is_valid_flip:
            return False, flip_reason

    return True, "Valid"

def format_grid(grid):
    """Format grid as string"""
    return "\n".join([" ".join([str(cell) for cell in row]) for row in grid])

def count_dsl_primitives(program: str) -> int:
    """
    Count number of primitives in DSL program
    Examples:
        (lambda (flipx $0)) -> 1
        (lambda (gravity_down (flipx $0))) -> 2
        (lambda (ic_fill (ic_compress2 (pickmax_interior_count (ic_splitall $0))))) -> 4
    """
    import re
    # Remove lambda
    clean = program.replace('lambda', '')
    # Find all primitives (function names after opening parenthesis)
    primitives = re.findall(r'\(([a-z_0-9]+)', clean)
    # Remove $ variables
    primitives = [p for p in primitives if not p.startswith('$') and not p.startswith('c')]
    return len(primitives)

def get_dsl_description(program: str) -> str:
    """Get natural language description of first DSL primitive found"""
    for prim, desc in COMPLETE_DSL_DESCRIPTIONS.items():
        if prim in program:
            return f"{prim}: {desc}"
    return ""

def get_all_dsl_descriptions(program: str) -> str:
    """Get descriptions for ALL primitives in the program"""
    import re
    # Find all primitives
    clean = program.replace('lambda', '')
    primitives = re.findall(r'\(([a-z_0-9]+)', clean)
    primitives = [p for p in primitives if not p.startswith('$') and not p.startswith('c')]

    # Get unique primitives in order
    seen = set()
    unique_prims = []
    for p in primitives:
        if p not in seen and p in COMPLETE_DSL_DESCRIPTIONS:
            seen.add(p)
            unique_prims.append(p)

    # Build description
    if not unique_prims:
        return ""

    desc_lines = []
    for i, prim in enumerate(unique_prims, 1):
        desc = COMPLETE_DSL_DESCRIPTIONS[prim]
        desc_lines.append(f"  {i}. {prim}: {desc}")

    return "\n".join(desc_lines)

def apply_dsl_program(program_str: str, input_grid: list) -> list:
    """Apply DSL program to input grid - simplified for common primitives"""
    try:
        input_array = np.array(input_grid)
        grid_obj = Grid(input_array)

        # Handle single primitives
        if 'rot270' in program_str:
            result = rot270(grid_obj)
        elif 'rot90' in program_str:
            result = rot90(grid_obj)
        elif 'rot180' in program_str:
            result = rot180(grid_obj)
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
        elif 'swapxy' in program_str:
            result = swapxy(grid_obj)
        elif 'mirrorX' in program_str:
            result = mirrorX(grid_obj)
        elif 'mirrorY' in program_str:
            result = mirrorY(grid_obj)
        else:
            return None

        return result.grid.tolist() if hasattr(result, 'grid') else result
    except Exception as e:
        return None

def create_type1_prompt_improved(sample_data: dict) -> str:
    """Create IMPROVED TYPE 1 prompt with CORRECTED examples and flexible step count"""

    # Get ALL DSL descriptions
    all_dsl_desc = get_all_dsl_descriptions(sample_data['program'])

    # Count primitives
    primitive_count = count_dsl_primitives(sample_data['program'])

    # Handle cases where there might be only 1 example
    example1 = sample_data['examples'][0]
    example2 = sample_data['examples'][1] if len(sample_data['examples']) > 1 else sample_data['examples'][0]

    prompt = f"""You are analyzing an ARC (Abstract Reasoning Corpus) puzzle transformation.

**Training Examples:**

Example 1 Input:
{format_grid(example1['input'])}

Example 1 Output:
{format_grid(example1['output'])}

Example 2 Input:
{format_grid(example2['input'])}

Example 2 Output:
{format_grid(example2['output'])}

**Test Input:**
{format_grid(sample_data['test'][0]['input'])}

**Test Output:**
{format_grid(sample_data['test'][0]['output'])}

**DSL Program:** {sample_data['program']}
**Primitive Count:** {primitive_count}
**DSL Primitives Used:**
{all_dsl_desc}

---

**EXAMPLE RESPONSE FORMATS** (showing proper channel usage):

**Example 1: flipx - IMPORTANT CORRECTED DEFINITION**
For (lambda (flipx $0)):

Input:  [[1, 2],
         [3, 4]]
Output: [[3, 4],   ← Bottom row moved to top
         [1, 2]]   ← Top row moved to bottom

<|channel|>analysis<|message|>
This is VERTICAL flip (upside-down). Swaps top and bottom rows.
<|end|><|start|>assistant<|channel|>final<|message|>
**Step 1:** Flip the grid vertically (upside-down), swapping the top and bottom rows while keeping each row's internal order.

The resulting grid is the vertical reflection of the original.
<|return|>

**Example 2: flipy - IMPORTANT CORRECTED DEFINITION**
For (lambda (flipy $0)):

Input:  [[1, 2],
         [3, 4]]
Output: [[2, 1],   ← Row 1 reversed
         [4, 3]]   ← Row 2 reversed

<|channel|>analysis<|message|>
This is HORIZONTAL flip (left-right). Swaps columns within each row.
<|end|><|start|>assistant<|channel|>final<|message|>
**Step 1:** Flip the grid horizontally (left-right), swapping the left and right sides while preserving row order.

The resulting grid is the horizontal reflection of the original.
<|return|>

**Example 3: gravity_left (2-step)**
For (lambda (gravity_left $0)):

<|channel|>analysis<|message|>
This moves colored cells left within each row. Two actions needed.
<|end|><|start|>assistant<|channel|>final<|message|>
**Step 1:** Shift each non-zero cell in every row as far left as possible, keeping the original order.
**Step 2:** Leave all other cells as black (0).

All colored cells are now packed to the left of each row.
<|return|>

---

**CRITICAL REQUIREMENTS:**
1. ALWAYS use <|channel|>analysis first for brief reasoning (1-2 sentences)
2. Then IMMEDIATELY switch to <|channel|>final<|message|> for your answer
3. In final channel, use **Step 1:**, **Step 2:** format (one sentence per step)
4. **NUMBER OF STEPS:**
   - Maximum steps = Primitive Count (number of DSL functions)
   - You MAY use fewer steps if some primitives don't need separate explanation
   - You do NOT need to explain every primitive separately if they form a simple concept
   - Focus on clarity - combine steps when it makes the explanation clearer
5. Provide ONE final summary sentence after all steps
6. Do NOT output grids or code in final channel
7. CRITICAL: flipx = VERTICAL flip (top↔bottom), flipy = HORIZONTAL flip (left↔right)

**STEP COUNT EXAMPLES:**
- Single primitive (count=1): Usually 1 step, sometimes 2 if complex
- Two primitives (count=2): 1-2 steps (can combine if closely related)
- Three+ primitives (count=3+): 2-4 steps (combine related operations)

Your task: Analyze the transformation above and provide your explanation following the EXACT format shown in the examples.
Remember: Primitive Count = {primitive_count}, so use UP TO {primitive_count} steps (fewer is fine if clearer)."""

    return prompt

def create_type2_prompt_improved(sample_data: dict, wrong_program: str, wrong_output: list) -> str:
    """Create IMPROVED TYPE 2 prompt with flexible step count"""

    # Get ALL DSL descriptions for correct program
    all_dsl_desc = get_all_dsl_descriptions(sample_data['program'])

    # Count primitives in correct program
    primitive_count = count_dsl_primitives(sample_data['program'])

    # Get description for wrong program
    wrong_dsl_desc = get_dsl_description(wrong_program)

    # Handle cases where there might be only 1 example
    example1 = sample_data['examples'][0]
    example2 = sample_data['examples'][1] if len(sample_data['examples']) > 1 else sample_data['examples'][0]

    prompt = f"""You are analyzing an ARC (Abstract Reasoning Corpus) puzzle transformation.

**Training Examples (showing CORRECT transformation):**

Example 1 Input:
{format_grid(example1['input'])}

Example 1 Output:
{format_grid(example1['output'])}

Example 2 Input:
{format_grid(example2['input'])}

Example 2 Output:
{format_grid(example2['output'])}

**Test Input:**
{format_grid(sample_data['test'][0]['input'])}

**INCORRECT Output (from wrong program {wrong_program}):**
{format_grid(wrong_output)}

**CORRECT Output (expected):**
{format_grid(sample_data['test'][0]['output'])}

**Wrong Program:** {wrong_program}
**Wrong DSL Description:** {wrong_dsl_desc}

**Correct Program:** {sample_data['program']}
**Correct Primitive Count:** {primitive_count}
**Correct DSL Primitives:**
{all_dsl_desc}

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
1. Start with **Incorrect transformation:** explaining what the wrong program did
2. Then provide corrected steps using **Step 1:**, **Step 2:** format
3. **NUMBER OF STEPS (for corrected version):**
   - Maximum steps = Correct Primitive Count
   - You MAY use fewer steps if some primitives don't need separate explanation
   - Focus on clarity - combine steps when it makes the explanation clearer
4. Provide ONE final summary sentence after all steps
5. Do NOT output grids or code
6. CRITICAL: flipx = VERTICAL flip (top↔bottom), flipy = HORIZONTAL flip (left↔right)

Your task: Explain why the incorrect output is wrong and what the correct steps should be.
Remember: Correct Primitive Count = {primitive_count}, so use UP TO {primitive_count} steps (fewer is fine if clearer)."""

    return prompt

def query_gpt_oss_with_retry(prompt: str, model, tokenizer, max_retries: int = 3, program: str = "") -> tuple:
    """Query GPT-OSS with retry logic and validation"""

    for attempt in range(max_retries):
        # Add stronger instruction on retries
        current_prompt = prompt
        if attempt > 0:
            current_prompt += f"\n\n**CRITICAL (RETRY {attempt})**: You MUST use <|channel|>final<|message|> and provide clear step-by-step explanation!"

        messages = [{"role": "user", "content": current_prompt}]

        # Apply chat template with reasoning_effort="low"
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="low"
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        try:
            # Generate (CRITICAL: use no_grad to prevent memory leak)
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=4096,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True
                )

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

            # Explicitly delete tensors to free memory
            del model_inputs
            del generated_ids
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            # Clean up and return OOM error status
            print(f"  ⚠ CUDA Out of Memory error: {str(e)[:100]}")
            if 'model_inputs' in locals():
                del model_inputs
            if 'generated_ids' in locals():
                del generated_ids
            torch.cuda.empty_cache()
            gc.collect()
            return "", 0, "oom_error"

        # Extract final channel
        final_text = extract_final_channel_improved(response)

        # Validate
        is_valid, reason = is_valid_response(response, final_text, program)

        if is_valid:
            return final_text, attempt + 1, "success"

        # If failed validation, try again
        print(f"  Attempt {attempt + 1} failed: {reason}")

    # All retries failed
    return final_text, max_retries, f"failed: {reason}"

def generate_wrong_programs(correct_program: str) -> list:
    """Generate plausible wrong programs"""
    wrong_programs = []

    # Extract main primitive
    main_prim = None
    for prim in COMPLETE_DSL_DESCRIPTIONS.keys():
        if prim in correct_program:
            main_prim = prim
            break

    if main_prim is None:
        return ['(lambda (flipx $0))']  # fallback

    # Strategy 1: Use opposite transformation
    opposites = {
        'gravity_left': 'gravity_right',
        'gravity_right': 'gravity_left',
        'gravity_up': 'gravity_down',
        'gravity_down': 'gravity_up',
        'flipx': 'flipy',
        'flipy': 'flipx',
        'rot90': 'rot270',
        'rot270': 'rot90',
    }

    if main_prim in opposites:
        wrong_programs.append(f'(lambda ({opposites[main_prim]} $0))')

    # Strategy 2: Use different common primitive
    if main_prim not in ['flipx', 'flipy']:
        wrong_programs.append('(lambda (flipx $0))')

    # Strategy 3: Use compose (multi-step wrong)
    if len(wrong_programs) > 0:
        return wrong_programs[:2]  # Return top 2

    return ['(lambda (flipx $0))']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID (1, 2, or 3)')
    parser.add_argument('--start', type=int, required=True, help='Start sample index')
    parser.add_argument('--end', type=int, required=True, help='End sample index')
    args = parser.parse_args()

    # Check for existing checkpoint directory (resume from OOM crash)
    existing_dir = f'/data/helmarc_gpt_analysis_v2/full_dataset_gpu{args.gpu}_20251025_154440'
    if os.path.exists(existing_dir):
        output_dir = existing_dir
        print(f"✓ Resuming from existing directory: {output_dir}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'/data/helmarc_gpt_analysis_v2/full_dataset_gpu{args.gpu}_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Created new directory: {output_dir}")

    print("="*80)
    print(f"FULL DATASET GENERATION V2 - GPU {args.gpu}")
    print("="*80)
    print(f"Samples: {args.start} to {args.end} ({args.end - args.start} samples)")
    print(f"Expected: {(args.end - args.start) * 2} total outputs (TYPE 1 + TYPE 2)")
    print(f"IMPROVEMENTS:")
    print(f"  - 67 complete DSL descriptions (vs 19 previously)")
    print(f"  - CORRECTED flipx/flipy definitions")
    print(f"  - Enhanced validation logic")
    print(f"  - Improved in-context examples")
    print("="*80)
    print()

    # Load model
    model_name = "openai/gpt-oss-20b"
    print(f"Loading GPT-OSS model on GPU {args.gpu}...")
    print(f"Using device: cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    print()

    # Load HelmARC data
    helmarc_path = '/data/helmarc_correct/20251024_062500/samples.json'
    with open(helmarc_path, 'r') as f:
        all_samples = json.load(f)

    samples = all_samples[args.start:args.end]
    print(f"Loaded {len(samples)} samples")
    print()

    # Results storage - load from checkpoint if exists
    type1_results = []
    type2_results = []
    skipped_samples = []  # Track OOM samples
    start_idx = 0

    # Find latest checkpoint (FIXED: sort by numeric value, not string)
    checkpoint_files = [f for f in os.listdir(output_dir) if f.startswith('type1_checkpoint_')]
    if checkpoint_files:
        # Extract numeric values and find maximum
        checkpoint_nums = [int(f.split('_')[-1].replace('.json', '')) for f in checkpoint_files]
        latest_checkpoint_num = max(checkpoint_nums)
        checkpoint_file1 = os.path.join(output_dir, f'type1_checkpoint_{latest_checkpoint_num}.json')
        checkpoint_file2 = os.path.join(output_dir, f'type2_checkpoint_{latest_checkpoint_num}.json')

        print(f"✓ Found checkpoint at sample {latest_checkpoint_num}")
        print(f"  Loading: {checkpoint_file1}")
        print(f"  Loading: {checkpoint_file2}")

        with open(checkpoint_file1, 'r') as f:
            type1_results = json.load(f)
        with open(checkpoint_file2, 'r') as f:
            type2_results = json.load(f)

        # Load skipped samples if exists
        skipped_file = os.path.join(output_dir, 'skipped_samples_oom.json')
        if os.path.exists(skipped_file):
            with open(skipped_file, 'r') as f:
                skipped_samples = json.load(f)

        start_idx = latest_checkpoint_num
        print(f"✓ Resuming from sample {start_idx} (already completed {len(type1_results)} samples)")
        if skipped_samples:
            print(f"✓ Previously skipped {len(skipped_samples)} samples due to OOM")
        print()
    else:
        print("✓ No checkpoint found, starting from beginning")
        print()

    # Process each sample
    for idx, sample in enumerate(samples):
        global_idx = args.start + idx

        # Skip already processed samples
        if idx < start_idx:
            continue

        print(f"\n[{global_idx+1}/{args.end}] Processing sample {global_idx}...")

        # TYPE 1: Correct DSL → Plan
        print("  TYPE 1: Generating correct DSL explanation...")
        type1_prompt = create_type1_prompt_improved(sample)
        type1_response, type1_attempts, type1_status = query_gpt_oss_with_retry(
            type1_prompt, model, tokenizer, program=sample['program']
        )

        # Check for OOM error
        if type1_status == "oom_error":
            print(f"  ⚠ SKIPPING sample {global_idx} due to OOM error")
            skipped_samples.append({
                'sample_index': global_idx,
                'task_id': sample['task_id'],
                'program': sample['program'],
                'reason': 'oom_error_type1'
            })
            # Save skipped samples immediately
            skipped_file = os.path.join(output_dir, 'skipped_samples_oom.json')
            with open(skipped_file, 'w') as f:
                json.dump(skipped_samples, f, indent=2)
            continue

        type1_results.append({
            'sample_index': global_idx,
            'task_id': sample['task_id'],
            'program': sample['program'],
            'final_response': type1_response,
            'attempts': type1_attempts,
            'status': type1_status
        })

        print(f"  TYPE 1: {type1_status} (attempts: {type1_attempts})")

        # TYPE 2: Wrong DSL → Correction
        print("  TYPE 2: Generating error correction...")
        wrong_programs = generate_wrong_programs(sample['program'])
        wrong_program = wrong_programs[0]

        # Apply wrong program
        try:
            wrong_output = apply_dsl_program(wrong_program, sample['test'][0]['input'])
            if wrong_output is None:
                wrong_output = sample['test'][0]['input']  # fallback
        except:
            wrong_output = sample['test'][0]['input']

        type2_prompt = create_type2_prompt_improved(sample, wrong_program, wrong_output)
        type2_response, type2_attempts, type2_status = query_gpt_oss_with_retry(
            type2_prompt, model, tokenizer
        )

        # Check for OOM error on TYPE 2
        if type2_status == "oom_error":
            print(f"  ⚠ SKIPPING TYPE 2 for sample {global_idx} due to OOM error")
            # Still keep TYPE 1 result but mark TYPE 2 as skipped
            type2_results.append({
                'sample_index': global_idx,
                'task_id': sample['task_id'],
                'correct_program': sample['program'],
                'wrong_program': wrong_program,
                'final_response': "",
                'attempts': 0,
                'status': 'oom_error'
            })
            skipped_samples.append({
                'sample_index': global_idx,
                'task_id': sample['task_id'],
                'program': sample['program'],
                'reason': 'oom_error_type2'
            })
            # Save skipped samples immediately
            skipped_file = os.path.join(output_dir, 'skipped_samples_oom.json')
            with open(skipped_file, 'w') as f:
                json.dump(skipped_samples, f, indent=2)
        else:
            type2_results.append({
                'sample_index': global_idx,
                'task_id': sample['task_id'],
                'correct_program': sample['program'],
                'wrong_program': wrong_program,
                'final_response': type2_response,
                'attempts': type2_attempts,
                'status': type2_status
            })

        print(f"  TYPE 2: {type2_status} (attempts: {type2_attempts})")

        # Clear GPU memory to prevent OOM (already cleared in query_gpt_oss_with_retry)
        # Additional cleanup for safety
        gc.collect()

        # Save checkpoint every 100 samples
        if (idx + 1) % 100 == 0:
            checkpoint_file1 = os.path.join(output_dir, f'type1_checkpoint_{idx+1}.json')
            checkpoint_file2 = os.path.join(output_dir, f'type2_checkpoint_{idx+1}.json')

            with open(checkpoint_file1, 'w') as f:
                json.dump(type1_results, f, indent=2)
            with open(checkpoint_file2, 'w') as f:
                json.dump(type2_results, f, indent=2)

            print(f"  ✓ Checkpoint saved at sample {idx+1}")

    # Save final results
    print("\n" + "="*80)
    print("Saving final results...")

    final_file1 = os.path.join(output_dir, 'type1_correct_dsl_to_plan.json')
    final_file2 = os.path.join(output_dir, 'type2_wrong_dsl_to_correction.json')

    with open(final_file1, 'w') as f:
        json.dump(type1_results, f, indent=2)
    with open(final_file2, 'w') as f:
        json.dump(type2_results, f, indent=2)

    # Save final skipped samples list
    if skipped_samples:
        skipped_file = os.path.join(output_dir, 'skipped_samples_oom.json')
        with open(skipped_file, 'w') as f:
            json.dump(skipped_samples, f, indent=2)

    # Statistics
    type1_success = sum(1 for x in type1_results if x['status'] == 'success')
    type2_success = sum(1 for x in type2_results if x['status'] == 'success')

    print(f"\n✓ TYPE 1: {type1_success}/{len(type1_results)} successful")
    print(f"✓ TYPE 2: {type2_success}/{len(type2_results)} successful")
    print(f"✓ TOTAL: {type1_success + type2_success}/{len(type1_results) + len(type2_results)} successful")
    if skipped_samples:
        print(f"⚠ SKIPPED: {len(skipped_samples)} samples due to OOM (see skipped_samples_oom.json)")
    print(f"\nOutput directory: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
