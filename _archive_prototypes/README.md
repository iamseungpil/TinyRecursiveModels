# Archive: Prototype Scripts and Outputs

This directory contains prototype scripts and outputs from the initial development phase of the HelmARC dataset generation project.

## Contents

### Prototype Generation Scripts
- **generate_10_samples.py** - Initial prototype for generating 10+10 TYPE 1 and TYPE 2 samples
  - Generated first test samples with basic prompt templates
  - Identified issues with max_new_tokens (1024) and temperature (0.7)
  - Success rate: 70% (14/20 samples)

- **generate_improved_10_samples.py** - Improved prototype with better parameters
  - Increased max_new_tokens to 4096
  - Reduced temperature to 0.3 for consistency
  - Added retry logic for failed generations
  - Success rate: 100% (20/20 samples)
  - **This version's improvements were incorporated into generate_full_dataset.py**

### Prototype Outputs
- **prototype_10_samples/** - Output from initial prototype
  - type1_correct_dsl_to_plan.json (10 samples)
  - type2_wrong_dsl_to_correction.json (10 samples)

- **prototype_10_samples_improved/** - Output from improved prototype
  - type1_correct_dsl_to_plan.json (10 samples)
  - type2_wrong_dsl_to_correction.json (10 samples)

### Documentation
- **PROTOTYPE_10_SAMPLES_SUMMARY.md** - Results from initial prototype run
- **IMPROVED_RESULTS_SUMMARY.md** - Results from improved prototype run showing 100% success rate

### Sample Selection Files
- **selected_10_samples.json** - 10 samples selected for initial testing (diverse primitives)
- **helmarc_sample_examples.json** - Sample metadata from HelmARC dataset

### Logs
- **generation_improved.log** - Generation log from improved prototype
- **incontext_generation.log** - Log from in-context example generation

## Historical Context

These prototypes were essential for:
1. Testing GPT-OSS channel switching behavior
2. Identifying optimal generation parameters
3. Developing the prompt template structure
4. Validating the TYPE 1 and TYPE 2 dataset format

The lessons learned from these prototypes directly informed the production implementation in **generate_full_dataset.py**, which successfully generated the full HelmARC dataset (16,549 samples).

## Superseded By

All functionality from these prototypes has been incorporated into:
- **/home/ubuntu/TinyRecursiveModels/generate_full_dataset.py** (production script)
- **/home/ubuntu/TinyRecursiveModels/launch_full_generation.sh** (production launcher)

## Date Archived

2025-10-25
