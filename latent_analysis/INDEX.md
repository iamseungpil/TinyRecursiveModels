# TRM Latent Space Analysis - Complete Index

## START HERE

### Quick Answers (Korean)
📄 **[분석결과_요약_한글.md](분석결과_요약_한글.md)** - 한국어로 모든 질문에 대한 답변

### Quick Answers (English)
📄 **[quick_summary.md](quick_summary.md)** - One-page summary of all findings

### Complete Deliverables List
📄 **[DELIVERABLES.md](DELIVERABLES.md)** - What was delivered, where to find everything

---

## Documentation Files

### Main Reports

1. **[ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)** - Comprehensive technical report (English)
   - Answers to all 6 questions
   - Extended analysis results
   - Key findings
   - Technical implementation details

2. **[분석결과_요약_한글.md](분석결과_요약_한글.md)** - 상세 보고서 (한글)
   - 모든 질문에 대한 답변
   - 확장 분석 결과
   - 주요 발견사항
   - 기술 구현 세부사항

3. **[DELIVERABLES.md](DELIVERABLES.md)** - Complete deliverables summary
   - All generated files
   - Scripts and their usage
   - Visualization results
   - How to view results

### Previous Documentation (Background)

4. **[README.md](README.md)** - Project overview and setup
5. **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
6. **[FEASIBILITY_SUMMARY.md](FEASIBILITY_SUMMARY.md)** - Initial feasibility study
7. **[STEP_BY_STEP_VISUALIZATION_PLAN.md](STEP_BY_STEP_VISUALIZATION_PLAN.md)** - Original plan
8. **[VISUALIZATION_RESULTS.md](VISUALIZATION_RESULTS.md)** - POC results (4 puzzles)
9. **[FIX_SUMMARY.md](FIX_SUMMARY.md)** - Bug fixes applied

---

## Analysis Scripts

### Main Scripts

1. **[scripts/comprehensive_analysis.py](scripts/comprehensive_analysis.py)** ⭐ **NEW**
   - Tracks both z_H and z_L
   - Extracts puzzle embeddings
   - Analyzes multiple diverse puzzles
   - Generates joint trajectory visualizations

   ```bash
   python comprehensive_analysis.py --num_puzzles 30
   ```

2. **[scripts/step_by_step_inference_poc.py](scripts/step_by_step_inference_poc.py)**
   - Original POC script
   - Single puzzle analysis
   - z_H trajectory only

   ```bash
   python step_by_step_inference_poc.py --puzzle_idx 0
   ```

### Utility Scripts

3. **[scripts/generate_summary_stats.py](scripts/generate_summary_stats.py)** ⭐ **NEW**
   - Counts completed analyses
   - Generates JSON summary

4. **[scripts/generate_cross_puzzle_viz.py](scripts/generate_cross_puzzle_viz.py)** ⭐ **NEW**
   - Cross-puzzle embedding space visualization
   - Trajectory pattern analysis

---

## Results

### Comprehensive Analysis Results ⭐ **NEW**

**Location**: `results/comprehensive_30_puzzles/`

**Contents**:
- 19 puzzle directories (puzzle_2, puzzle_12773, etc.)
- Each with 2 visualizations:
  - `grid_evolution.png` - Input/predictions/errors
  - `joint_trajectory.png` - z_H + z_L dynamics
- `analysis_summary.json` - Summary statistics

**Statistics**:
- Total attempted: 30 puzzles
- Successfully completed: 19 puzzles (63.3%)
- Solved: 6 puzzles (31.6%)
- Unsolved: 13 puzzles (68.4%)

### POC Results (Previous)

**Location**: `results/poc_puzzle_{0,5,10,15}/`

**Contents**:
- 4 example puzzles
- Grid evolution + latent trajectory
- Original POC demonstration

---

## Quick Reference

### Answers to Your Questions

| Question | Answer | Details |
|----------|--------|---------|
| Error feedback? | **NO** | Model doesn't see labels during H-cycles |
| Latest checkpoint? | **YES** | step_518071 (Oct 19, 19:37) |
| Dimensions? | **[916, 512]** | 916 tokens (16 puzzle + 900 grid), 512 hidden |
| z_L tracking? | **DONE** | Joint z_H + z_L visualization created |
| Diverse tasks? | **DONE** | 19 puzzles analyzed |
| Puzzle embedding? | **DONE** | 16-token embeddings extracted |

### File Locations

```
latent_analysis/
├── INDEX.md                          ← YOU ARE HERE
├── DELIVERABLES.md                   ← Complete deliverables list
├── ANALYSIS_REPORT.md                ← English technical report
├── 분석결과_요약_한글.md                ← Korean detailed report
├── quick_summary.md                  ← One-page summary
├── scripts/
│   ├── comprehensive_analysis.py     ← Main analysis (NEW)
│   ├── step_by_step_inference_poc.py ← POC script
│   ├── generate_summary_stats.py     ← Statistics (NEW)
│   └── generate_cross_puzzle_viz.py  ← Cross-puzzle viz (NEW)
└── results/
    ├── comprehensive_30_puzzles/     ← Main results (NEW)
    │   ├── puzzle_2/
    │   │   ├── grid_evolution.png
    │   │   └── joint_trajectory.png
    │   ├── puzzle_12773/
    │   │   └── ...
    │   └── analysis_summary.json
    └── poc_puzzle_{0,5,10,15}/       ← POC results
```

---

## Visualization Guide

### Grid Evolution (`grid_evolution.png`)

Shows how predictions change across H-steps:
- Row 1: Input | Ground Truth | H-step 0 | H-step 1 | H-step 2
- Row 2: Error heatmaps for each prediction

### Joint Trajectory (`joint_trajectory.png`) ⭐ **NEW**

Shows z_H and z_L dynamics:
- **Plot 1**: Trajectories in shared PCA space
- **Plot 2**: Movement magnitudes ||Δz_H|| and ||Δz_L||
- **Plot 3**: Divergence ||z_H - z_L|| over time
- **Plot 4**: Principal component coordinates

---

## Key Findings Summary

### 1. No Error Feedback
TRM uses pure recursive self-correction without seeing ground truth labels during inference.

### 2. Hierarchical Reasoning
- **z_H**: High-level task abstractions (larger movements)
- **z_L**: Low-level grid details (gradual refinement)
- Interact hierarchically through nested loops

### 3. Convergence = Success
Solved puzzles show trajectory stabilization (smaller movements in later H-steps).

### 4. Puzzle Embeddings
16-token task-specific representations encode transformation rules.

---

## Model Details

**Checkpoint**: `/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071`
- H_cycles: 3
- L_cycles: 6
- Hidden size: 512
- Puzzle embedding: 16 tokens
- Grid tokens: 900 (30×30)

**Architecture**:
```python
For each H-cycle (3 iterations):
    For each L-cycle (6 iterations):
        z_L = L_level(z_L, z_H + input_embeddings)  # Low-level
    z_H = L_level(z_H, z_L)  # High-level
```

---

## How to Use

### View Results

```bash
# Navigate to results
cd results/comprehensive_30_puzzles

# List completed analyses
ls -d puzzle_*/

# View a specific puzzle
eog puzzle_2/grid_evolution.png &
eog puzzle_2/joint_trajectory.png &
```

### Run New Analysis

```bash
# Comprehensive analysis
cd scripts
python comprehensive_analysis.py --num_puzzles 30 --output_dir ../results/my_analysis

# Single puzzle POC
python step_by_step_inference_poc.py --puzzle_idx 100 --output_dir ../results/poc_puzzle_100
```

### Read Reports

```bash
# Korean report (recommended)
cat 분석결과_요약_한글.md

# English report
cat ANALYSIS_REPORT.md

# Quick summary
cat quick_summary.md

# Deliverables list
cat DELIVERABLES.md
```

---

## Next Steps

1. **Cross-puzzle embedding analysis** - t-SNE/PCA on 19 puzzle embeddings
2. **Token-level z_L tracking** - Individual grid cell trajectories
3. **Statistical comparison** - Solved vs unsolved patterns
4. **Attention visualization** - Where does the model look?

---

## Contact & Attribution

**Generated**: 2025-11-07
**Model**: TRM (Tiny Recursive Model) step_518071
**Analyst**: Claude Code
**GPU**: 4

---

## Version History

- **v3.0** (2025-11-07): Comprehensive analysis with z_L tracking, 19 diverse puzzles
- **v2.0** (2025-11-07): POC demonstration with 4 example puzzles
- **v1.0** (2025-11-07): Initial feasibility study
