# Solution Recommendations for Improving Success Rate

Generated on: 2025-10-24

---

## Current Status

- **TYPE 1**: 6/10 success (60%)
- **TYPE 2**: 8/10 success (80%)
- **Overall**: 14/20 samples training-ready (70%)

**Root Cause**: Model doesn't always switch to `<|channel|>final` (NOT a chat template issue)

---

## Recommended Solution (Option A): Improve Prompts + Regenerate Failed Samples

### Step 1: Update Prompt Template

**Add explicit final channel instruction**:

```python
def create_type1_prompt_improved(sample_data: dict) -> str:
    prompt = f"""You are analyzing an ARC (Abstract Reasoning Corpus) puzzle transformation.

**Training Examples:**
[examples as before]

**Transformation Program:** {sample_data['program']}
**DSL Information:** {dsl_desc}

**Example Response Format:**

For (lambda (flipx $0)):

<|channel|>analysis<|message|>
This is horizontal mirroring. Only one step needed.
<|end|><|start|>assistant<|channel|>final<|message|>
**Step 1:** Mirror the entire grid horizontally, swapping left and right sides.

This is a single-step geometric transformation.
<|return|>

---

**IMPORTANT FORMAT REQUIREMENTS**:
1. First use <|channel|>analysis to briefly plan (1-2 sentences)
2. Then ALWAYS switch to <|channel|>final for your answer
3. In final channel, use **Step 1:**, **Step 2:** format
4. Provide 3-4 sentences maximum
5. Do NOT output grids or code

Your task: Provide a CONCISE explanation following the exact format above."""

    return prompt
```

**Key improvements**:
- Shows explicit example with channel switching
- Instructs model to use analysis THEN final
- Emphasizes "ALWAYS switch to <|channel|>final"

### Step 2: Increase Generation Capacity

```python
# OLD
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024,
    temperature=0.7,
    do_sample=True
)

# NEW (Recommended)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048,      # Double the capacity
    temperature=0.3,          # More deterministic
    do_sample=True,
    top_p=0.9                 # Add nucleus sampling
)
```

**Why**:
- More tokens for both analysis (500) + final (300-500)
- Lower temperature = more consistent channel switching
- top_p=0.9 reduces low-probability outputs

### Step 3: Add Multiple In-Context Examples

**Current**: 1 example (flipx)
**Recommended**: 3 examples (1-step, 2-step, 3-step)

```python
EXAMPLE_1_STEP = """
For (lambda (flipx $0)):

<|channel|>final<|message|>
**Step 1:** Mirror the entire grid horizontally, swapping left and right sides.

This is a single-step geometric transformation.
"""

EXAMPLE_2_STEP = """
For (lambda (gravity_left $0)):

<|channel|>final<|message|>
**Step 1:** Shift all non-zero cells in each row as far left as possible.
**Step 2:** Preserve the original order of cells within each row.

All colored cells are now packed to the left.
"""

EXAMPLE_3_STEP = """
For (lambda (ic_center $0)):

<|channel|>final<|message|>
**Step 1:** Identify the smallest bounding box around all non-zero cells.
**Step 2:** Calculate the center position of the grid.
**Step 3:** Shift the bounding box to center it horizontally and vertically.

The pattern is now centered with zeros surrounding it.
"""
```

### Step 4: Implement Retry Logic

```python
def generate_with_retry(prompt: str, model, tokenizer, max_retries=2):
    """Generate with automatic retry for failed channel switching"""
    for attempt in range(max_retries):
        response = query_gpt_oss(prompt, model, tokenizer)
        final = extract_final_channel(response)

        # Validate response
        is_valid = (
            '<|channel|>final' in response and
            '**Step 1:**' in final and
            len(final) > 100 and
            len(final) < 800
        )

        if is_valid:
            print(f"✓ Valid response on attempt {attempt + 1}")
            return response

        # If failed, add more explicit instruction
        if attempt < max_retries - 1:
            print(f"⚠️ Retry {attempt + 2}: Response invalid")
            prompt += "\n\n**CRITICAL**: You MUST use <|channel|>final<|message|> for your answer!"

    print(f"❌ Failed after {max_retries} attempts")
    return response  # Return last attempt even if failed
```

### Step 5: Regenerate Only Failed Samples

**Failed TYPE 1 samples** (4 samples):
1. Sample 1: ic_connectX
2. Sample 5: mirrorX (truncated)
3. Sample 6: ic_erasecol
4. Sample 7: ic_fill

**Failed TYPE 2 samples** (2 samples):
1. Sample 8: gravity_down
2. Sample 9: flipy

**Total**: 6 samples to regenerate

---

## Expected Improvements

### After Implementing All Steps:

| Change | Expected Impact |
|--------|----------------|
| Explicit channel instruction | +15% success rate |
| max_new_tokens=2048 | +5% (fixes truncation) |
| temperature=0.3 | +5% (more consistent) |
| Multiple in-context examples | +10% success rate |
| Retry logic | +5% (auto-recovery) |

**Expected new success rate**: **85-95%** (17-19 / 20 samples)

### Scaling to Full Dataset:

With improved prompts + retry logic:
- Current 8,572 HelmARC samples
- Expected success: **~85%** = **7,286 samples**
- Processing time: ~167 sec/sample × 8,572 = **~400 hours** (16-17 days on single GPU)

**Optimization**: Use 3 GPUs in parallel = **~5-6 days**

---

## Alternative Solutions

### Option B: Accept 70% and Scale Now

**Pros**:
- Faster start (no prompt changes needed)
- Still get 6,000+ samples (70% of 8,572)
- Can refine prompts during scaling

**Cons**:
- Lower quality (30% failure rate)
- Need post-processing filtering
- May need second pass later

**Recommendation**: Not recommended - 15% improvement is worth the effort

### Option C: Hybrid Approach

1. **Scale with current prompts** for 1,000 samples (test run)
2. **Analyze results** to identify patterns
3. **Refine prompts** based on actual failures
4. **Scale remaining 7,572 samples** with improved prompts

**Pros**:
- Learn from larger sample size
- Data-driven prompt optimization
- Less risk of over-fitting to 10 samples

**Cons**:
- Slower (two-phase approach)
- May waste GPU time on first 1,000

**Recommendation**: Good compromise if uncertain about Option A

---

## Implementation Plan (Recommended: Option A)

### Phase 1: Improve and Validate (1-2 hours)

1. ✅ Update `generate_10_samples.py` with improved prompts
2. ✅ Increase max_new_tokens to 2048, temperature to 0.3
3. ✅ Add 3 in-context examples (1-step, 2-step, 3-step)
4. ✅ Implement retry logic
5. ✅ Test on 6 failed samples

**Success criteria**: 5-6 / 6 samples successful (83-100%)

### Phase 2: Scale to Full Dataset (5-6 days on 3 GPUs)

1. ✅ Split 8,572 samples across GPU 1, 2, 3
2. ✅ Run generation with improved prompts
3. ✅ Monitor success rate every 100 samples
4. ✅ Adjust prompts if success rate < 80%

**Success criteria**: 7,000+ successful samples (>80%)

### Phase 3: Quality Control (1 day)

1. ✅ Filter samples with quality checks:
   - Has `<|channel|>final` marker
   - Has `**Step 1:**` pattern
   - Length 100-800 chars
   - No grid output in response
2. ✅ Manual review of 100 random samples
3. ✅ Create final training dataset

**Success criteria**: 7,000+ high-quality samples ready for training

---

## Cost-Benefit Analysis

### Option A: Improve Now (Recommended)

- **Time investment**: 1-2 hours prompt engineering + 5-6 days generation
- **Expected output**: 7,286 samples (85% × 8,572)
- **Quality**: High (85%+ success rate)
- **Cost**: ~400 GPU hours (distributed across 3 GPUs)

**ROI**: 1,286 more samples (7,286 vs 6,000) = +21% dataset size for 2 hours work

### Option B: Accept 70% and Scale

- **Time investment**: 0 hours (immediate start) + 5-6 days generation
- **Expected output**: 6,000 samples (70% × 8,572)
- **Quality**: Medium (70% success rate)
- **Cost**: ~400 GPU hours (distributed across 3 GPUs)

**ROI**: Start immediately but lose 1,286 potential samples

### Verdict

**Option A is better**: 2 hours of prompt engineering yields 21% more data.

---

## Next Steps

**If you approve Option A**:

```bash
# 1. Create improved generation script
cp generate_10_samples.py generate_improved.py

# 2. Test on 6 failed samples
python3 generate_improved.py --test-mode --samples-to-test 6

# 3. If successful, scale to full dataset
python3 generate_improved.py --full-dataset --gpus 1,2,3
```

**Expected completion**: 5-7 days
**Expected output**: 7,000+ high-quality TYPE 1 + TYPE 2 samples
