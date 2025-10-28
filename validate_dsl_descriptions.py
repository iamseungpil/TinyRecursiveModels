#!/usr/bin/env python3
"""
Validate DSL descriptions against reference definitions
"""

import json
import re

# DSL Reference Definitions (from DSL_PRIMITIVES_REFERENCE.md)
DSL_DEFINITIONS = {
    # Rotation
    'rot90': 'Rotates the grid 90 degrees clockwise',
    'rot180': 'Rotates the grid 180 degrees',
    'rot270': 'Rotates the grid 270 degrees clockwise (or 90 degrees counter-clockwise)',
    'swapxy': 'Transposes the grid, swapping rows and columns (diagonal flip)',

    # Flip - CORRECTED DEFINITIONS
    'flipx': 'Flips the grid VERTICALLY (upside-down), swapping top and bottom rows',
    'flipy': 'Flips the grid HORIZONTALLY (left-right), swapping left and right columns',

    # Mirror (expand)
    'mirrorX': 'Extends the grid by mirroring HORIZONTALLY (creates left-right symmetry)',
    'mirrorY': 'Extends the grid by mirroring VERTICALLY (creates top-bottom symmetry)',

    # Gravity
    'gravity_left': 'Moves all colored cells to the left within each row',
    'gravity_right': 'Moves all colored cells to the right within each row',
    'gravity_up': 'Moves all colored cells upward within each column',
    'gravity_down': 'Moves all colored cells downward within each column',

    # Half selection
    'top_half': 'Returns the top half of the grid',
    'bottom_half': 'Returns the bottom half of the grid',
    'left_half': 'Returns the left half of the grid',
    'right_half': 'Returns the right half of the grid',

    # IC operations
    'ic_fill': 'Fills interior empty cells within colored regions',
    'ic_compress2': 'Removes empty rows and columns to compress the grid (level 2)',
    'ic_compress3': 'Removes empty rows and columns to compress the grid (level 3)',
    'ic_erasecol': 'Removes all cells of a specific color (replaces with black/0)',
    'ic_filtercol': 'Filters cells by color, keeping only specified colors',
    'ic_interior': 'Extracts only the interior cells of colored regions',
    'ic_invert': 'Inverts the color pattern',
    'ic_makeborder': 'Creates a border around colored regions',
    'ic_toorigin': 'Moves all colored cells to the origin (top-left)',
    'ic_embed': 'Embeds one grid pattern into another',
    'ic_connectX': 'Connects separated components of the same color horizontally',
    'ic_connectY': 'Connects separated components of the same color vertically',
    'ic_splitall': 'Splits the grid into all connected components',
    'ic_splitcols': 'Splits by color groups',
    'ic_splitcolumns': 'Splits the grid by columns into separate subgrids',
    'ic_splitrows': 'Splits the grid by rows into separate subgrids',

    # Pick operations
    'pickmax_size': 'Selects the largest component by size',
    'pickmax_count': 'Selects the component with maximum count',
    'pickmax_x_pos': 'Selects component with maximum x position',
    'pickmax_x_neg': 'Selects component with minimum x position',
    'pickmax_y_pos': 'Selects component with maximum y position',
    'pickmax_y_neg': 'Selects component with minimum y position',
    'pickmax_neg_size': 'Selects component with maximum negative size metric',

    # Color operations
    'setcol': 'Sets all cells to a specific color',
    'set_bg': 'Sets the background color to a specified value',
    'get_bg': 'Identifies and returns cells matching the background color',
    'rarestcol': 'Finds the rarest (least frequent) color in the grid',
    'topcol': 'Identifies the color of the topmost non-zero cell',

    # Composition
    'overlay': 'Overlays one grid pattern on top of another',
    'fillobj': 'Fills objects with a specific pattern',
    'logical_and': 'Performs logical AND operation between two grids',
}

def extract_primitives(program):
    """Extract all DSL primitives from program"""
    content = program.replace('lambda', '').replace('$0', '').replace('$1', '')
    primitives = re.findall(r'\b([a-z_]+\d*)(?=\s|\()', content)
    # Filter out color constants
    return [p for p in primitives if not p.startswith('c') or p in ['count', 'colour', 'colourHull']]

def check_flipx_description(response):
    """Check if flipx is described correctly"""
    issues = []
    if 'flipx' in response.lower() or 'flip' in response.lower():
        # Should mention "vertical" or "upside-down" or "top" and "bottom"
        has_vertical = 'vertical' in response.lower() or 'upside' in response.lower()
        has_topbottom = 'top' in response.lower() and 'bottom' in response.lower()

        # Should NOT mention "horizontal" or "left-right" for flipx
        has_horizontal = 'horizontal' in response.lower() and 'left' in response.lower() and 'right' in response.lower()

        if has_horizontal and not (has_vertical or has_topbottom):
            issues.append("flipx described as HORIZONTAL (WRONG - should be VERTICAL)")
        elif not (has_vertical or has_topbottom) and 'flip' in response.lower():
            issues.append("flipx missing vertical/upside-down description")

    return issues

def check_flipy_description(response):
    """Check if flipy is described correctly"""
    issues = []
    if 'flipy' in response.lower() or 'flip' in response.lower():
        # Should mention "horizontal" or "left-right" or "left" and "right"
        has_horizontal = 'horizontal' in response.lower() or ('left' in response.lower() and 'right' in response.lower())

        # Should NOT mention "vertical" or "top-bottom" for flipy
        has_vertical = 'vertical' in response.lower() and 'top' in response.lower() and 'bottom' in response.lower()

        if has_vertical and not has_horizontal:
            issues.append("flipy described as VERTICAL (WRONG - should be HORIZONTAL)")
        elif not has_horizontal and 'flip' in response.lower():
            issues.append("flipy missing horizontal/left-right description")

    return issues

def check_gravity_description(response, direction):
    """Check if gravity direction is described correctly"""
    issues = []
    direction_map = {
        'gravity_left': ['left', 'leftward'],
        'gravity_right': ['right', 'rightward'],
        'gravity_up': ['up', 'upward', 'top'],
        'gravity_down': ['down', 'downward', 'bottom']
    }

    expected_keywords = direction_map.get(direction, [])
    if direction in response.lower():
        # Check if any expected keyword is present
        has_direction = any(keyword in response.lower() for keyword in expected_keywords)
        if not has_direction:
            issues.append(f"{direction} missing direction keyword ({', '.join(expected_keywords)})")

    return issues

def check_rotation_description(response, rot_type):
    """Check if rotation is described correctly"""
    issues = []
    rotation_map = {
        'rot90': ['90', 'clockwise', '90°'],
        'rot180': ['180', '180°'],
        'rot270': ['270', '90', 'counter-clockwise', 'counterclockwise']
    }

    expected_keywords = rotation_map.get(rot_type, [])
    if rot_type in response.lower():
        has_rotation = any(keyword in response.lower() for keyword in expected_keywords)
        if not has_rotation:
            issues.append(f"{rot_type} missing rotation description")

    return issues

def validate_sample(sample):
    """Validate a single sample's description"""
    program = sample['program']
    response = sample['final_response']
    primitives = extract_primitives(program)

    issues = []

    # Check flipx/flipy
    if 'flipx' in primitives:
        issues.extend(check_flipx_description(response))
    if 'flipy' in primitives:
        issues.extend(check_flipy_description(response))

    # Check gravity operations
    for prim in primitives:
        if prim.startswith('gravity_'):
            issues.extend(check_gravity_description(response, prim))

    # Check rotations
    for prim in primitives:
        if prim in ['rot90', 'rot180', 'rot270']:
            issues.extend(check_rotation_description(response, prim))

    # Check if step count roughly matches primitive count
    steps = re.findall(r'\*\*Step (\d+)\*\*', response)
    step_count = len(steps)
    primitive_count = len(primitives)

    if step_count > 0:
        step_ratio = abs(step_count - primitive_count) / primitive_count if primitive_count > 0 else 0
        if step_ratio > 0.5:  # More than 50% difference
            issues.append(f"Step count mismatch: {step_count} steps for {primitive_count} primitives (ratio: {step_ratio:.1%})")

    return {
        'sample_index': sample['sample_index'],
        'primitives': primitives,
        'primitive_count': len(primitives),
        'step_count': step_count,
        'issues': issues
    }

def main():
    print("="*70)
    print("DSL DESCRIPTION VALIDATION - Complex Programs")
    print("="*70)

    # Load the top 10 complex samples we analyzed
    gpu1_data = json.load(open('/data/helmarc_gpt_analysis_v2/full_dataset_gpu1_20251025_154440/type1_correct_dsl_to_plan.json'))
    gpu2_data = json.load(open('/data/helmarc_gpt_analysis_v2/full_dataset_gpu2_20251025_154440/type1_correct_dsl_to_plan.json'))
    gpu3_data = json.load(open('/data/helmarc_gpt_analysis_v2/full_dataset_gpu3_20251025_154440/type1_checkpoint_2800.json'))

    all_data = gpu1_data + gpu2_data + gpu3_data
    successful = [item for item in all_data if item['status'] == 'success']

    # Count primitives
    for item in successful:
        item['primitive_count'] = len(extract_primitives(item['program']))

    # Sort by complexity
    successful_sorted = sorted(successful, key=lambda x: x['primitive_count'], reverse=True)

    # Validate top 20 complex samples
    print(f"\nValidating top 20 most complex programs...")
    print("="*70)

    validation_results = []
    for i, sample in enumerate(successful_sorted[:20], 1):
        result = validate_sample(sample)
        validation_results.append(result)

        print(f"\n--- Sample {i} ---")
        print(f"Index: {result['sample_index']}")
        print(f"Primitive count: {result['primitive_count']}")
        print(f"Step count: {result['step_count']}")
        print(f"Primitives: {', '.join(result['primitives'][:10])}{'...' if len(result['primitives']) > 10 else ''}")

        if result['issues']:
            print(f"⚠ Issues found ({len(result['issues'])}):")
            for issue in result['issues']:
                print(f"  - {issue}")
        else:
            print("✓ No issues detected")

    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")

    total_issues = sum(len(r['issues']) for r in validation_results)
    samples_with_issues = sum(1 for r in validation_results if r['issues'])

    print(f"\nTotal samples validated: 20")
    print(f"Samples with issues: {samples_with_issues}/20 ({100*samples_with_issues/20:.1f}%)")
    print(f"Total issues found: {total_issues}")

    if samples_with_issues > 0:
        print(f"\n⚠ Issue breakdown:")
        issue_types = {}
        for r in validation_results:
            for issue in r['issues']:
                issue_type = issue.split(':')[0] if ':' in issue else issue
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1

        for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {issue_type}: {count}")

    # Additional check: Sample some medium complexity programs (5-8 primitives)
    print(f"\n{'='*70}")
    print("ADDITIONAL VALIDATION - Medium Complexity Programs")
    print(f"{'='*70}")

    medium_complex = [s for s in successful if 5 <= s['primitive_count'] <= 8]
    print(f"\nTotal medium-complexity programs (5-8 primitives): {len(medium_complex)}")

    # Sample 10 random medium-complexity programs
    import random
    random.seed(42)
    medium_sample = random.sample(medium_complex, min(10, len(medium_complex)))

    medium_validation = []
    for i, sample in enumerate(medium_sample, 1):
        result = validate_sample(sample)
        medium_validation.append(result)

        print(f"\n--- Medium Sample {i} ---")
        print(f"Index: {result['sample_index']}")
        print(f"Primitives ({result['primitive_count']}): {', '.join(result['primitives'])}")
        print(f"Steps: {result['step_count']}")

        if result['issues']:
            print(f"⚠ Issues: {', '.join(result['issues'])}")
        else:
            print("✓ No issues")

    medium_issues = sum(len(r['issues']) for r in medium_validation)
    medium_with_issues = sum(1 for r in medium_validation if r['issues'])

    print(f"\n{'='*70}")
    print("MEDIUM COMPLEXITY SUMMARY")
    print(f"{'='*70}")
    print(f"Samples with issues: {medium_with_issues}/10")
    print(f"Total issues: {medium_issues}")

if __name__ == "__main__":
    main()
