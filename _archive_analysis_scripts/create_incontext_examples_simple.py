#!/usr/bin/env python3
"""
Generate in-context examples for two new dataset types:
1. Correct DSL → Plan explanation (step count + natural language)
2. Wrong DSL → Error explanation (why the wrong DSL doesn't match)

This simplified version uses the GPT-OSS model directly.
"""

import json
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add paths for DSL primitives
sys.path.insert(0, '/home/ubuntu/dreamcoder-arc/ec')
from dreamcoder.domains.arc.arcPrimitivesIC2 import rot270, ic_connectX, Grid

def format_grid_for_prompt(grid):
    """Format a grid as a readable string."""
    return "\n".join([" ".join([str(cell) for cell in row]) for row in grid])

def query_gpt_oss(prompt, model, tokenizer):
    """Query GPT-OSS model with a prompt."""
    # Format with harmony response format
    messages = [{"role": "user", "content": prompt}]

    # Tokenize
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True
    )

    # Decode
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def create_type1_example(model, tokenizer):
    """TYPE 1: Correct DSL → Plan explanation"""
    print("\n" + "="*80)
    print("TYPE 1 EXAMPLE: Correct DSL → Plan Explanation")
    print("="*80)

    # Load the ic_connectX sample
    with open('/data/helmarc_correct/20251024_062500/samples.json', 'r') as f:
        data = json.load(f)

    sample = None
    for s in data:
        if s['task_id'] == 'helmholtz_99b1bc43_1120':
            sample = s
            break

    if not sample:
        print("ERROR: Sample not found")
        return

    # Create the prompt
    prompt_type1 = f"""You are analyzing an ARC (Abstract Reasoning Corpus) puzzle transformation.

**Training Examples:**

Example 1 Input:
{format_grid_for_prompt(sample['examples'][0]['input'])}

Example 1 Output:
{format_grid_for_prompt(sample['examples'][0]['output'])}

Example 2 Input:
{format_grid_for_prompt(sample['examples'][1]['input'])}

Example 2 Output:
{format_grid_for_prompt(sample['examples'][1]['output'])}

**Test Input:**
{format_grid_for_prompt(sample['test'][0]['input'])}

**Correct Transformation Program:** {sample['program']}

Your task is to provide a CONCISE analysis (approximately 3 sentences) that includes:
1. How many transformation steps are required for this problem
2. A natural language explanation of what the transformation does

Keep your response brief and focused. Do NOT output grids or code."""

    print("\n" + "-"*80)
    print("PROMPT FOR TYPE 1:")
    print("-"*80)
    print(prompt_type1)
    print("\n" + "-"*80)
    print("Calling GPT-OSS with reasoning_effort=low...")
    print("-"*80)

    try:
        response = query_gpt_oss(prompt_type1, model, tokenizer)

        print("\n" + "-"*80)
        print("GPT-OSS RESPONSE (TYPE 1):")
        print("-"*80)
        print(response)
        print("\n")

        # Save to file
        output = {
            "type": "correct_dsl_to_plan",
            "task_id": sample['task_id'],
            "program": sample['program'],
            "prompt": prompt_type1,
            "response": response,
            "reasoning_effort": "low"
        }

        with open('/home/ubuntu/TinyRecursiveModels/helmarc_visualization/incontext_type1.json', 'w') as f:
            json.dump(output, f, indent=2)

        print("✓ Saved to helmarc_visualization/incontext_type1.json")

    except Exception as e:
        print(f"ERROR calling GPT-OSS: {e}")
        import traceback
        traceback.print_exc()

def create_type2_example(model, tokenizer):
    """TYPE 2: Wrong DSL → Error explanation"""
    print("\n" + "="*80)
    print("TYPE 2 EXAMPLE: Wrong DSL → Error Explanation")
    print("="*80)

    # Load the ic_connectX sample
    with open('/data/helmarc_correct/20251024_062500/samples.json', 'r') as f:
        data = json.load(f)

    sample = None
    for s in data:
        if s['task_id'] == 'helmholtz_99b1bc43_1120':
            sample = s
            break

    if not sample:
        print("ERROR: Sample not found")
        return

    # Apply WRONG DSL (rot270 instead of ic_connectX)
    wrong_dsl = "(lambda (rot270 $0))"
    test_input = sample['test'][0]['input']
    test_input_array = np.array(test_input)  # Convert list to numpy array
    test_input_grid = Grid(test_input_array)  # Create Grid object
    wrong_output_grid = rot270(test_input_grid)
    wrong_output = wrong_output_grid.grid.tolist()  # Extract grid as list
    correct_output = sample['test'][0]['output']

    # Create the prompt
    prompt_type2 = f"""You are analyzing an ARC (Abstract Reasoning Corpus) puzzle transformation.

**Training Examples (showing CORRECT transformation):**

Example 1 Input:
{format_grid_for_prompt(sample['examples'][0]['input'])}

Example 1 Output:
{format_grid_for_prompt(sample['examples'][0]['output'])}

Example 2 Input:
{format_grid_for_prompt(sample['examples'][1]['input'])}

Example 2 Output:
{format_grid_for_prompt(sample['examples'][1]['output'])}

**Test Input:**
{format_grid_for_prompt(test_input)}

**INCORRECT Output (from wrong transformation {wrong_dsl}):**
{format_grid_for_prompt(wrong_output)}

**CORRECT Output (expected):**
{format_grid_for_prompt(correct_output)}

Your task is to provide a CONCISE explanation (approximately 3 sentences) of:
1. Why the incorrect output does NOT match the pattern from the training examples
2. What is fundamentally wrong with the transformation that was applied

Keep your response brief and focused. Do NOT output grids or code."""

    print("\n" + "-"*80)
    print("PROMPT FOR TYPE 2:")
    print("-"*80)
    print(prompt_type2)
    print("\n" + "-"*80)
    print("Calling GPT-OSS with reasoning_effort=low...")
    print("-"*80)

    try:
        response = query_gpt_oss(prompt_type2, model, tokenizer)

        print("\n" + "-"*80)
        print("GPT-OSS RESPONSE (TYPE 2):")
        print("-"*80)
        print(response)
        print("\n")

        # Save to file
        output = {
            "type": "wrong_dsl_to_error",
            "task_id": sample['task_id'],
            "correct_program": sample['program'],
            "wrong_program": wrong_dsl,
            "test_input": test_input,
            "wrong_output": wrong_output,
            "correct_output": correct_output,
            "prompt": prompt_type2,
            "response": response,
            "reasoning_effort": "low"
        }

        with open('/home/ubuntu/TinyRecursiveModels/helmarc_visualization/incontext_type2.json', 'w') as f:
            json.dump(output, f, indent=2)

        print("✓ Saved to helmarc_visualization/incontext_type2.json")

    except Exception as e:
        print(f"ERROR calling GPT-OSS: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
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

    print("Model loaded successfully!")

    print("\nGenerating in-context examples for two new dataset types...")
    print("This will call GPT-OSS twice with reasoning_effort=low")

    # Create both examples
    create_type1_example(model, tokenizer)
    create_type2_example(model, tokenizer)

    print("\n" + "="*80)
    print("DONE! Both in-context examples generated.")
    print("="*80)
    print("\nPlease review:")
    print("  1. helmarc_visualization/incontext_type1.json (Correct DSL → Plan)")
    print("  2. helmarc_visualization/incontext_type2.json (Wrong DSL → Error)")
    print("\nIf approved, I will proceed with full implementation.")
