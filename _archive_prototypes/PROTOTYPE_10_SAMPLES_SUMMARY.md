# Prototype 10+10 Samples Generation Summary

Generated on: 2025-10-24
Model: GPT-OSS-20B (openai/gpt-oss-20b)
Device: CUDA GPU 1

---

## Overview

Generated **10 TYPE 1** (Correct DSL → Plan) and **10 TYPE 2** (Wrong DSL → Correction) samples using diverse DSL primitives.

---

## TYPE 1: Correct DSL → Step-by-Step Plan

**Format**: Given correct DSL program, generate step-by-step plan with **Step N:** format

| # | Primitive | Task ID | Status | Final Response Length |
|---|-----------|---------|--------|----------------------|
| 1 | ic_connectX | helmholtz_99b1bc43_1120 | ⚠️ Partial | 499 chars (analysis leak) |
| 2 | rot270 | helmholtz_39e1d7f9_5808 | ✅ Success | 325 chars |
| 3 | gravity_left | helmholtz_82819916_2096 | ✅ Success | 250 chars |
| 4 | flipx | helmholtz_d9f24cd1_6224 | ✅ Success | 180 chars |
| 5 | mirrorX | helmholtz_99fa7670_3824 | ⚠️ Truncated | 33 chars |
| 6 | ic_erasecol | helmholtz_6aa20dc0_4320 | ⚠️ Partial | 499 chars (analysis leak) |
| 7 | ic_fill | helmholtz_150deff5_7392 | ⚠️ Partial | 499 chars (analysis leak) |
| 8 | gravity_down | helmholtz_3906de3d_7376 | ✅ Success | 266 chars |
| 9 | flipy | helmholtz_f9012d9b_8048 | ✅ Success | 217 chars |
| 10 | ic_center | helmholtz_53b68214_3216 | ✅ Success | 414 chars (2 steps!) |

**Success Rate**: 6/10 (60%)

### Successful Examples:

#### Sample 2 (rot270) ✅
```
**Step 1:** Rotate the entire grid 270 degrees clockwise (equivalent to a 90‑degree counter‑clockwise rotation).
The resulting grid is the original pattern turned so that its former bottom becomes the new left side, its former left becomes the new top, and so on.
The output grid is the input grid rotated 270° clockwise.
```
**Analysis**: Perfect 3-sentence format with Step 1: and summary.

#### Sample 3 (gravity_left) ✅
```
**Step 1:** Shift each non‑zero cell in every row as far left as possible, keeping the original order of the cells within that row.
**Step 2:** No other modifications are made to the grid.
All colored cells are now packed to the left of each row.
```
**Analysis**: Perfect 3-sentence format with 2 steps and summary.

#### Sample 4 (flipx) ✅
```
**Step 1:** Mirror the entire grid horizontally, swapping left and right sides while preserving all cell colors.

The resulting grid is the horizontal reflection of the original.
```
**Analysis**: Perfect 2-sentence format, concise and clear.

#### Sample 10 (ic_center) ✅
```
**Step 1:** Identify the smallest rectangle that encloses all non‑zero cells in the grid.
**Step 2:** Shift that bounding rectangle so that it is centered horizontally and vertically within the grid dimensions.
**Step 3:** All other cells remain zero.

The resulting grid has the pattern centered, with surrounding zeros.
```
**Analysis**: **Excellent** - 4 sentences with 3 steps! Exactly what we wanted.

---

## TYPE 2: Wrong DSL → Corrected Step-by-Step Plan

**Format**: Given wrong DSL applied, explain error and provide correct steps

| # | Correct Primitive | Wrong DSL | Status | Final Response Length |
|---|-------------------|-----------|--------|----------------------|
| 1 | ic_connectX | rot270 | ✅ Success | 447 chars |
| 2 | rot270 | flipx | ✅ Success | 603 chars |
| 3 | gravity_left | gravity_right | ✅ Success | 471 chars |
| 4 | flipx | flipy | ✅ Success | 503 chars |
| 5 | mirrorX | flipx | ✅ Success | 545 chars |
| 6 | ic_erasecol | gravity_left | ✅ Success | 356 chars |
| 7 | ic_fill | ic_center | ✅ Success | 616 chars |
| 8 | gravity_down | compose flipx gravity_left | ⚠️ Partial | 500 chars |
| 9 | flipy | compose rot270 ic_fill | ⚠️ Partial | 500 chars |
| 10 | ic_center | compose ic_center flipy gravity_down | ✅ Success | 762 chars |

**Success Rate**: 8/10 (80%)

### Successful Examples:

#### Sample 1 (ic_connectX vs rot270) ✅
```
**Incorrect transformation:** the program rotated the entire grid 270° clockwise, producing a rotated pattern instead of merging blocks.
**Step 1:** locate each color that appears in two separate horizontal regions of the same row...
```
**Analysis**: Good format with **Incorrect transformation:** and **Step 1:**.

#### Sample 3 (gravity_left vs gravity_right) ✅
```
**Incorrect transformation:** it applied gravity_right, moving all non‑zero cells to the far right of each row.
**Step 1:** Move each non‑zero cell in every row to the far left...
```
**Analysis**: Clear explanation of wrong vs correct transformation.

#### Sample 10 (ic_center vs 3-step compose) ✅
```
**Incorrect transformation:** The gravity_down rule pulled every colored cell to the bottom of its column, leaving almost all rows empty and producing a grid of zeros.

**Step 1:** Locate the smallest bounding box...
```
**Analysis**: **Excellent** - Multi-step wrong DSL (3-step compose) explained correctly!

---

## Key Findings

### ✅ **Successes**:

1. **Step Format**: 6/10 TYPE 1 and 8/10 TYPE 2 successfully generated **Step N:** format
2. **Concise Responses**: Most successful samples are 150-600 chars (3-4 sentences)
3. **Multi-Step**: Sample 10 (ic_center) generated **3 steps** naturally!
4. **Wrong DSL Diversity**: TYPE 2 successfully handled:
   - 1-step wrong primitives (samples 1-7)
   - 2-step compose wrong DSL (sample 8-9)
   - 3-step compose wrong DSL (sample 10)

### ⚠️ **Issues**:

1. **Analysis Leakage**: 4/10 TYPE 1 samples leaked analysis reasoning into final response
2. **Channel Parsing**: Some samples didn't properly switch to `<|channel|>final`
3. **Truncation**: Sample 5 (mirrorX) was truncated to 33 chars

### 💡 **Why Success Rate Differs**:

- **TYPE 2 (80%) > TYPE 1 (60%)**: TYPE 2 prompts are more explicit about format
  - TYPE 2 explicitly says "**Incorrect transformation:**" and "**Step 1:**"
  - TYPE 1 relies more on in-context example

---

## Recommended Improvements

### For Full Production:

1. **Add More In-Context Examples** to TYPE 1 prompt:
   - Show 2-3 examples instead of 1
   - Include multi-step example (like ic_center)

2. **Explicit Format Instructions**:
   ```
   IMPORTANT: Your response MUST follow this exact format:
   **Step 1:** [description]
   **Step 2:** [description] (if needed)
   [Final summary sentence]
   ```

3. **Post-Processing Filter**:
   - Detect when analysis leaks into final response
   - Automatically retry with more explicit prompts

4. **Channel Switching Emphasis**:
   - Add to system prompt: "Always use <|channel|>final for your final answer"
   - Possibly inject `<|channel|>final<|message|>` in prompt

---

## Files Generated

- `prototype_10_samples/type1_correct_dsl_to_plan.json` - TYPE 1 dataset (10 samples)
- `prototype_10_samples/type2_wrong_dsl_to_correction.json` - TYPE 2 dataset (10 samples)
- `selected_10_samples.json` - Selected diverse primitives
- `PROTOTYPE_10_SAMPLES_SUMMARY.md` - This summary

---

## Next Steps

1. ✅ **Prototype Complete**: 10+10 samples generated with 60-80% success rate
2. ⏭️ **Decision Point**:
   - **Option A**: Improve prompts and regenerate failed samples
   - **Option B**: Accept 60-80% success rate and scale to full 8,572 samples with post-processing
3. ⏭️ **Scale to Full Dataset**: Apply to all HelmARC samples with improved prompts

---

## Sample Quality Assessment

### Excellent Samples (Ready for Training):
- Sample 2, 3, 4, 8, 9, 10 (TYPE 1)
- Sample 1, 2, 3, 4, 5, 6, 7, 10 (TYPE 2)

### Need Regeneration:
- Sample 1, 5, 6, 7 (TYPE 1)
- Sample 8, 9 (TYPE 2)

**Overall**: **14/20 (70%) samples are training-ready!**
