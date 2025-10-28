# Chat Template Issue Analysis

## Question: "왜 저렇게 될까? chat template의 문제일까?"

Generated on: 2025-10-24

---

## Root Cause Analysis

### What's Actually Happening

Looking at the raw GPT-OSS responses from our 10+10 prototype samples, I found **there is NO "assistantfinal" text issue**. The model is correctly generating harmony response format channel markers in most cases.

### Evidence from Raw Responses

#### ✅ **Successful Samples (Sample 2: rot270)**
```
<|channel|>analysis<|message|>We need to explain transformation...
<|end|><|start|>assistant<|channel|>final<|message|>**Step 1:** Rotate the entire grid 270 degrees clockwise...
<|return|>
```

**Analysis**:
- Model correctly uses `<|channel|>analysis` for reasoning
- Model correctly switches to `<|channel|>final` for answer
- Channel markers are proper special tokens (not text)
- Extraction works perfectly

#### ⚠️ **Failed Samples (Sample 1: ic_connectX)**
```
<|channel|>analysis<|message|>We need to produce explanation of transformation for test input...
[3370 chars of analysis reasoning]
[NO <|channel|>final switch ever happens]
```

**Analysis**:
- Model correctly uses `<|channel|>analysis`
- Model NEVER switches to `<|channel|>final`
- Response ends while still in analysis channel
- Our extraction fallback takes last 500 chars (still analysis content)

#### 🔍 **Truncated Sample (Sample 5: mirrorX)**
```
<|channel|>analysis<|message|>...[detailed analysis 2,869 chars]
<|end|><|start|>assistant<|channel|>final<|message|>**Step 1:** Duplicate each row of
```

**Analysis**:
- Model correctly switched to final channel
- Response was truncated mid-sentence (max_new_tokens limit?)
- Extraction got incomplete response

---

## Real Problem: Not Chat Template, But Channel Switching Behavior

### Issue #1: Model Doesn't Always Switch to Final Channel (40% of TYPE 1)

**Why it happens**:
1. **Prompt not explicit enough**: TYPE 1 prompt doesn't explicitly instruct model to use `<|channel|>final`
2. **Model learned to stay in analysis**: During pretraining, model may have learned that some tasks don't need final channel
3. **Complex reasoning tasks**: Samples with complex DSL (ic_connectX, ic_erasecol, ic_fill) tend to stay in analysis

**Evidence**: TYPE 2 succeeded 80% vs TYPE 1 60% because TYPE 2 prompt is more explicit about providing "final answer" format

### Issue #2: Generation Truncation (10% of samples)

**Why it happens**:
1. **max_new_tokens=1024**: Some analysis takes 500+ tokens, leaving little room for final response
2. **Long analysis**: Model spends too many tokens reasoning in analysis channel
3. **No stop criteria**: Model doesn't know when to stop analysis and switch to final

**Example**: Sample 5 (mirrorX) spent 2,869 chars on analysis, then only had 33 chars left for final

### Issue #3: Analysis Leakage (When extraction fails)

**Why it happens**:
1. **No final channel**: When model doesn't switch, our extraction has no `<|channel|>final` to find
2. **Fallback logic**: We fall back to "last 500 chars" which is still in analysis
3. **Pattern matching fails**: We try to find `**Step 1:**` pattern but analysis doesn't have it

---

## Why Chat Template is NOT the Problem

### What `apply_chat_template()` Does

```python
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

**Result**: Transforms messages into GPT-OSS format:
```
<|start|>user<|message|>Your task...
<|end|><|start|>assistant
```

**This is working correctly!** Evidence: All samples show proper `<|start|>` and `<|end|>` markers.

### What Chat Template Does NOT Control

Chat template does NOT control:
1. Whether model uses `<|channel|>analysis` or `<|channel|>final`
2. When model switches between channels
3. How much reasoning model does in each channel

**These are model generation behaviors**, not chat template formatting issues.

---

## Why TYPE 2 Succeeded More (80% vs 60%)

### TYPE 2 Prompt Structure (More Explicit)

```
Your task: Provide a CONCISE explanation (3-4 sentences maximum) that includes:
1. What the incorrect transformation did (one sentence)
2. The correct steps in **Step N:** format
3. A final summary of the difference
```

**Key difference**:
- Explicitly asks for "explanation" (implies final answer)
- Clear 3-part structure guides model
- "Incorrect transformation:" provides strong anchor

### TYPE 1 Prompt Structure (Less Explicit)

```
Your task: Provide a CONCISE explanation (3-4 sentences maximum) that includes:
1. Each transformation step in **Step N:** format
2. A final summary sentence
```

**Key difference**:
- Less explicit about what constitutes "final answer"
- Complex DSL analysis may tempt model to stay in analysis channel
- No strong anchor like "Incorrect transformation:"

---

## Solutions

### Solution 1: Explicit Final Channel Injection (Recommended)

**Modify prompt to include**:
```python
prompt += """

**IMPORTANT**: You MUST provide your final answer in the following format:

<|channel|>final<|message|>
**Step 1:** [description]
**Step 2:** [description] (if needed)
[Final summary sentence]
<|return|>
"""
```

**Expected improvement**: 80-90% success rate by explicitly showing expected format

### Solution 2: Increase max_new_tokens

**Current**: `max_new_tokens=1024`
**Recommended**: `max_new_tokens=2048`

**Reasoning**: Gives model more room for both analysis and final response

### Solution 3: Add More In-Context Examples

**Current**: 1 example (flipx)
**Recommended**: 2-3 examples showing variety of step counts

**Example additions**:
- 1-step example (flipx) ✓ already have
- 2-step example (gravity_left: Step 1 + Step 2)
- 3-step example (ic_center: Step 1 + Step 2 + Step 3)

### Solution 4: Post-Processing with Retry Logic

```python
def generate_with_retry(prompt, model, tokenizer, max_retries=2):
    for attempt in range(max_retries):
        response = query_gpt_oss(prompt, model, tokenizer)
        final = extract_final_channel(response)

        # Check if valid response
        if '**Step 1:**' in final and len(final) > 100:
            return response

        # If failed, add more explicit instruction
        if attempt < max_retries - 1:
            prompt += "\n\nIMPORTANT: Provide your final answer starting with **Step 1:**"

    return response  # Return last attempt even if failed
```

### Solution 5: Use Lower Temperature for Consistency

**Current**: `temperature=0.7`
**Recommended**: `temperature=0.3` or `temperature=0.5`

**Reasoning**: Less randomness may lead to more consistent channel switching behavior

---

## Comparison Table: Chat Template vs Generation Behavior

| Aspect | Chat Template Controls | Model Generation Controls |
|--------|------------------------|--------------------------|
| Message formatting | ✓ Yes (`<|start|>user`) | ✗ No |
| Role markers | ✓ Yes (`user`, `assistant`) | ✗ No |
| Channel usage | ✗ No | ✓ Yes (`analysis`, `final`) |
| Channel switching | ✗ No | ✓ Yes (when to switch) |
| Response length | ✗ No | ✓ Yes (depends on reasoning) |
| Format adherence | ✗ No | ✓ Yes (`**Step 1:**` format) |

---

## Recommended Next Steps

### Option A: Improve Prompts and Regenerate Failed Samples (Recommended)

1. **Add explicit final channel instruction** to TYPE 1 prompts
2. **Increase max_new_tokens** to 2048
3. **Add 2-3 in-context examples** showing multi-step transformations
4. **Lower temperature** to 0.3 for consistency
5. **Regenerate only failed 6 samples** (Samples 1, 5, 6, 7 from TYPE 1)

**Expected outcome**: 85-90% success rate (17-18 / 20 samples)

### Option B: Accept 70% Success Rate and Scale

1. **Use current 14 successful samples** as seed data
2. **Scale to full 8,572 samples** with improved prompts
3. **Post-process with automatic quality filtering**:
   - Keep samples with `**Step 1:**` pattern
   - Keep samples > 100 chars
   - Keep samples with final channel detected
4. **Expect ~70-80% success rate** (6,000-7,000 samples)

**Expected outcome**: 6,000-7,000 high-quality training samples

### Option C: Implement Retry Logic and Scale

1. **Implement Solution 4** (retry logic)
2. **Combine with Solution 1** (explicit instruction)
3. **Scale to full 8,572 samples**
4. **Expect 85%+ success rate** (~7,300 samples)

**Expected outcome**: 7,000+ high-quality training samples with automatic recovery

---

## Conclusion

**Answer to "왜 저렇게 될까? chat template의 문제일까?"**:

**No, it's NOT a chat template problem.**

The chat template is working correctly - all samples show proper message formatting with `<|start|>user<|message|>` and `<|start|>assistant` markers.

**The real issue is model generation behavior**:
1. Model doesn't always switch to `<|channel|>final` (especially for complex DSL tasks)
2. Model spends too many tokens in analysis channel, leaving little room for final response
3. Prompt is not explicit enough about requiring final channel usage

**The solution is better prompt engineering**, not fixing chat template:
- Add explicit final channel instructions
- Provide more in-context examples
- Increase max_new_tokens
- Lower temperature for consistency

**Evidence**: TYPE 2 (80% success) vs TYPE 1 (60% success) shows that more explicit prompts work better with the same chat template.
