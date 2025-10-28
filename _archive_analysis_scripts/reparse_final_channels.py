#!/usr/bin/env python3
"""
Re-parse final channels from TYPE 1 and TYPE 2 results
"""

import json
import re

def extract_final_channel_improved(response: str) -> str:
    """Improved final channel extraction"""
    # Method 1: Look for <|channel|>final<|message|>...<|return|>
    pattern1 = r'<\|channel\|>final<\|message\>(.*?)<\|return\|>'
    match1 = re.search(pattern1, response, re.DOTALL)
    if match1:
        content = match1.group(1).strip()
        if len(content) > 10:  # Valid content
            return content

    # Method 2: Look for <|channel|>final...<|return|> (includes <|message|>)
    pattern2 = r'<\|channel\|>final(.*?)<\|return\|>'
    match2 = re.search(pattern2, response, re.DOTALL)
    if match2:
        content = match2.group(1).strip()
        # Remove <|message|> if present
        content = re.sub(r'<\|message\|>', '', content).strip()
        if len(content) > 10:
            return content

    # Method 3: Look for text after "assistant" and "final"
    pattern3 = r'assistant.*?final.*?<\|message\|>(.*?)(?:<\|return\|>|$)'
    match3 = re.search(pattern3, response, re.DOTALL)
    if match3:
        content = match3.group(1).strip()
        if len(content) > 10:
            return content

    # Method 4: Last resort - look for **Step 1:** pattern after "final"
    if 'final' in response.lower():
        parts = response.split('final')
        if len(parts) > 1:
            last_part = parts[-1]
            # Find **Step 1:** or similar
            step_match = re.search(r'\*\*Step 1:\*\*(.*?)(?:<\|return\|>|$)', last_part, re.DOTALL)
            if step_match:
                # Get full content starting from **Step 1:**
                step_pos = last_part.find('**Step 1:**')
                if step_pos >= 0:
                    content = last_part[step_pos:]
                    # Remove trailing tags
                    content = re.sub(r'<\|return\|>.*$', '', content, flags=re.DOTALL)
                    return content.strip()

    # Fallback: return last 500 chars if too long
    if len(response) > 500:
        return response[-500:].strip()

    return response.strip()

def main():
    print("Re-parsing final channels from generated samples...")

    # Load TYPE 1
    with open('/home/ubuntu/TinyRecursiveModels/prototype_10_samples/type1_correct_dsl_to_plan.json', 'r') as f:
        type1_data = json.load(f)

    # Load TYPE 2
    with open('/home/ubuntu/TinyRecursiveModels/prototype_10_samples/type2_wrong_dsl_to_correction.json', 'r') as f:
        type2_data = json.load(f)

    print(f"\nProcessing {len(type1_data)} TYPE 1 samples...")
    for item in type1_data:
        old_final = item['final_response']
        new_final = extract_final_channel_improved(item['raw_response'])
        item['final_response'] = new_final

        print(f"\nSample {item['sample_index']} ({item['primitive']}):")
        print(f"  Old length: {len(old_final)} chars")
        print(f"  New length: {len(new_final)} chars")
        print(f"  Preview: {new_final[:100]}...")

    print(f"\nProcessing {len(type2_data)} TYPE 2 samples...")
    for item in type2_data:
        old_final = item['final_response']
        new_final = extract_final_channel_improved(item['raw_response'])
        item['final_response'] = new_final

        print(f"\nSample {item['sample_index']} ({item['primitive']}):")
        print(f"  Old length: {len(old_final)} chars")
        print(f"  New length: {len(new_final)} chars")
        print(f"  Preview: {new_final[:100]}...")

    # Save updated results
    with open('/home/ubuntu/TinyRecursiveModels/prototype_10_samples/type1_correct_dsl_to_plan.json', 'w') as f:
        json.dump(type1_data, f, indent=2)

    with open('/home/ubuntu/TinyRecursiveModels/prototype_10_samples/type2_wrong_dsl_to_correction.json', 'w') as f:
        json.dump(type2_data, f, indent=2)

    print(f"\n{'='*80}")
    print("✓ Re-parsing complete! Updated files saved.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
