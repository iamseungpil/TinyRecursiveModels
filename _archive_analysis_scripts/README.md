# Archive: One-Time Analysis Scripts

This directory contains one-time use analysis and debugging scripts from the HelmARC dataset generation and validation phase.

## Contents

### GPT-OSS Analysis Scripts
- **gpt_oss_analyzer.py** - GPT-OSS ARC analyzer class
  - Used for analyzing ARC puzzles with GPT-OSS model
  - Implemented channel switching investigation
  - Provided foundation for dataset generation approach

- **batch_analyze_helmarc.py** - Batch analyzer for HelmARC samples
  - Processed HelmARC samples in batches across GPUs
  - Used during initial dataset validation
  - Superseded by merge_gpt_analysis_results.py

- **start_gpt_oss_analysis.sh** - Launch script for batch analysis
  - Started analysis on GPUs 0-3
  - Used for distributed analysis of 8,572 samples
  - One-time use for initial dataset validation

### In-Context Example Generation
- **create_incontext_examples.py** - Generate in-context examples (complex version)
  - Created TYPE 1 and TYPE 2 in-context examples
  - Used GPT-OSS analyzer class
  - Tested different prompt formats

- **create_incontext_examples_simple.py** - Simplified in-context example generator
  - Direct GPT-OSS model usage
  - Generated examples for prompt development
  - Results incorporated into final prompt templates

### Parsing and Debugging Scripts
- **reparse_final_channels.py** - Re-parse final channels from GPT-OSS output
  - Developed improved channel extraction patterns
  - Tested multiple regex patterns for robustness
  - Parsing logic incorporated into generate_full_dataset.py

### Visualization Scripts
- **visualize_helmarc_sample.py** - Visualize individual HelmARC samples
  - Created grid visualizations with ARC color palette
  - Used for manual inspection during development
  - Superseded by visualize_helmarc_analysis.py (production)

- **analyze_trm_arc_results.py** - Analyze TRM evaluation results
  - Analyzed TRM checkpoint predictions on ARC validation set
  - One-time evaluation of step 518071
  - Part of TRM model evaluation (different project)

- **eval_and_visualize_trm.py** - TRM checkpoint evaluation and visualization
  - Loaded TRM checkpoint and generated predictions
  - Created visualizations of solved/unsolved problems
  - One-time model evaluation script

## Documentation Archived

### Analysis Documentation
- **CHAT_TEMPLATE_ANALYSIS.md** - Analysis of GPT-OSS chat template behavior
- **CHECKPOINT_ANALYSIS_SUMMARY.md** - TRM checkpoint analysis results
- **DSL_COVERAGE_ISSUE.md** - DSL primitive coverage issues identified
- **GPT_OSS_ANALYSIS_EXPLANATION.md** - Explanation of GPT-OSS analysis approach
- **INCONTEXT_EXAMPLES_SUMMARY.md** - Summary of in-context example generation
- **SOLUTION_RECOMMENDATIONS.md** - Recommendations for dataset generation
- **TRM_EVALUATION_SUMMARY.md** - TRM model evaluation summary

## Key Findings Preserved

1. **Channel Switching Behavior**: GPT-OSS requires explicit prompting for final channel
2. **Optimal Parameters**: max_new_tokens=4096, temperature=0.3
3. **Parsing Patterns**: Multiple fallback patterns needed for robust extraction
4. **In-Context Examples**: 1-3 examples improve generation quality

## Production Scripts (Still Active)

These scripts remain in the main directory for production use:
- **generate_full_dataset.py** - Production dataset generation
- **merge_gpt_analysis_results.py** - Merge analysis results
- **visualize_helmarc_analysis.py** - Production visualization tool

## Date Archived

2025-10-25
