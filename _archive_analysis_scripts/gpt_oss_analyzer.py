"""
GPT-OSS ARC Analyzer
Handles ARC analysis using GPT-OSS model with grid data (no images)
Based on sequential_validation_v4.py approach
"""
import os
import sys
import time
import torch
import logging
import numpy as np
from typing import Dict, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add GPT-OSS path for model loading
sys.path.append('/home/ubuntu/gpt_oss_arc_final')

# Import utils - use absolute import instead of relative
try:
    from utils import parse_step_response
except ImportError:
    # Fallback if utils not in path
    import sys
    sys.path.insert(0, '/home/ubuntu/dreamcoder-arc/InternVL/src')
    from utils import parse_step_response

class GPTOSSARCAnalyzer:
    """GPT-OSS based ARC puzzle analyzer"""

    def __init__(self, model_path: str = "openai/gpt-oss-20b", device: Optional[str] = None):
        """
        Initialize GPT-OSS analyzer

        Args:
            model_path: Path to GPT-OSS model
            device: Device to use (auto-detect if None)
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.device = device or self._get_device()

        self.logger.info(f"Initializing GPT-OSS analyzer with model: {model_path}")
        self.logger.info(f"Using device: {self.device}")

        # Initialize model and tokenizer
        self._load_model()

    def _get_device(self) -> str:
        """Get optimal device for inference"""
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"

    def _load_model(self):
        """Load GPT-OSS model and tokenizer"""
        try:
            self.logger.info("Loading GPT-OSS model...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False
            )

            # Load model - FORCE single GPU by using device string directly
            # device_map={"": device} doesn't work - transformers still distributes
            # Solution: Load to CPU first, then move to specific GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=None,  # Don't use auto device_map
                trust_remote_code=True,
                low_cpu_mem_usage=False  # Need to load to specific device
            )
            # Move entire model to specific GPU
            self.model = self.model.to(self.device)

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.logger.info("GPT-OSS model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load GPT-OSS model: {e}")
            raise

    def grid_to_string(self, grid: np.ndarray) -> str:
        """Convert numpy grid to string format"""
        return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

    def get_transformation_hint(self, program: str) -> str:
        """
        Extract transformation hint from DSL program without revealing the DSL itself
        """
        dsl_semantics = {
            "gravity_left": "The transformation moves all colored (non-zero) cells to the left within their row, compressing them together while preserving their original left-to-right order.",
            "gravity_right": "The transformation moves all colored (non-zero) cells to the right within their row, compressing them together while preserving their original left-to-right order.",
            "gravity_up": "The transformation moves all colored (non-zero) cells upward within their column, compressing them together while preserving their original top-to-bottom order.",
            "gravity_down": "The transformation moves all colored (non-zero) cells downward within their column, compressing them together while preserving their original top-to-bottom order.",
            "ic_erasecol": "The transformation replaces all cells of a specific color value with black (zero).",
            "flipx": "The transformation mirrors the grid horizontally (flips left-right).",
            "flipy": "The transformation mirrors the grid vertically (flips top-bottom).",
            "rot90": "The transformation rotates the entire grid 90 degrees clockwise.",
            "rot180": "The transformation rotates the entire grid 180 degrees.",
            "rot270": "The transformation rotates the entire grid 270 degrees clockwise (equivalent to 90 degrees counter-clockwise).",
        }

        # Extract operation name from program like "(lambda (gravity_left $0))"
        for op_name, hint in dsl_semantics.items():
            if op_name in program:
                return hint

        return "The transformation changes the grid according to a consistent pattern. Observe the examples carefully to identify the rule."

    def create_guided_prompt(self, helmholtz_data: Dict, program: str, expected_output: str) -> List[Dict]:
        """
        Create guided prompt for GPT-OSS analysis
        Following sequential_validation_v4.py format exactly
        """
        # Build training examples
        examples = []
        for i, example in enumerate(helmholtz_data.get('examples', []), 1):
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            examples.append(
                f"Example {i}:\n"
                f"Input:\n{self.grid_to_string(input_grid)}\n"
                f"Output:\n{self.grid_to_string(output_grid)}"
            )

        examples_text = '\n\n'.join(examples)

        # Get test input
        test_input = np.array(helmholtz_data['test'][0]['input'])

        # Get transformation hint
        transformation_hint = self.get_transformation_hint(program)

        # In-context example (REASONING ONLY - NO GRID OUTPUT)
        incontext_example = """<|channel|>analysis<|message|>We have two examples. Let me analyze each one carefully.

Example 1: Input is a 6x10 grid. Looking at the input row by row:
- Row 0: has a 2 at column 1 and an 8 at column 5
- Rows 1-5: all zeros

Output for Example 1:
- Row 0: has 2 at column 0 and 8 at column 1
- Rows 1-5: all zeros

So the 2 moved from column 1 to column 0 (shifted left by 1), and the 8 moved from column 5 to column 1 (shifted left by 4 columns). The non-zero values compressed to the left while maintaining their left-to-right order: 2 comes before 8 in the input, and 2 is still before 8 in the output.

Example 2: Input is a 6x7 grid.
- Row 0: has a 4 at column 1
- Rows 1-5: all zeros

Output for Example 2:
- Row 0: has 4 at column 0
- Rows 1-5: all zeros

The 4 moved from column 1 to column 0.

Pattern identified: For each row independently, all non-zero (colored) cells move to the leftmost positions of that row. They compress together, maintaining their original left-to-right sequence. The remaining positions in the row fill with zeros.

Now let me apply this to the test input. The test input has non-zero values in row 0: there's a 3 at column 2, a 6 at column 6, and a 7 at column 9. Other rows are all zeros.

Applying the rule:
- Row 0: Take the non-zero values (3, 6, 7) and place them at the leftmost columns (0, 1, 2), maintaining order. Fill the rest with zeros.
- Rows 1-5: All zeros, so they remain unchanged.<|end|><|start|>assistant<|channel|>final<|message|>**Detailed step-by-step reasoning**

**Step 1: Understanding the transformation pattern**
By examining both training examples, I observe that this transformation operates independently on each row of the grid. The key insight is that colored (non-zero) cells within a row are shifted to occupy the leftmost positions while preserving their original left-to-right ordering.

**Step 2: Identifying which cells are affected**
In the test input, I need to locate all non-zero cells:
- Row 0 contains three colored cells: the number 3 at column position 2, the number 6 at column position 6, and the number 7 at column position 9
- Rows 1 through 5 contain only zeros (black cells)

**Step 3: Preserving the sequential order**
The transformation maintains the relative ordering of colored cells as they appear from left to right in the original input. In row 0, reading from left to right, we encounter: first 3, then 6, then 7. This sequence 3→6→7 must be preserved in the output.

**Step 4: Applying the leftward compression**
The colored cells are moved to the leftmost available positions:
- The first colored cell (3) moves to column 0 (the leftmost position)
- The second colored cell (6) moves to column 1 (immediately adjacent)
- The third colored cell (7) moves to column 2 (immediately adjacent)
- All remaining positions in row 0 (columns 3 through 11) are filled with zeros

**Step 5: Handling rows with only zeros**
Rows 1, 2, 3, 4, and 5 contain no colored cells in the input. Since there are no non-zero values to move, these rows remain completely unchanged - they stay as rows of all zeros.

**Step 6: Verification of the transformation**
The transformation can be verified by confirming:
- Original colored cell order preserved: 3, 6, 7 (✓)
- Colored cells occupy leftmost positions: columns 0, 1, 2 (✓)
- No gaps between colored cells: they are consecutive (✓)
- Remaining cells filled with zeros: columns 3-11 all zero (✓)
- Unaffected rows remain unchanged: rows 1-5 stay all zeros (✓)<|return|>"""

        # Create messages following sequential_validation_v4.py format
        messages = [
            {
                "role": "system",
                "content": """You are an ARC Puzzle Solver.

Your purpose is to explain the reasoning behind ARC (Abstract Reasoning Corpus) transformations through detailed step-by-step analysis. Generate high-quality training data by thoroughly explaining how you identify patterns and apply transformation rules.

# DSL Primitive Functions Reference

These transformations may use combinations of the following operations:

## Geometric Transformations
- rot90: Rotate grid 90 degrees clockwise
- rot180: Rotate grid 180 degrees
- rot270: Rotate grid 270 degrees clockwise (90 degrees counterclockwise)
- flipx: Flip grid vertically (top-bottom mirror, along horizontal axis)
- flipy: Flip grid horizontally (left-right mirror, along vertical axis)
- swapxy: Transpose grid (swap rows and columns)
- identity: Return grid unchanged

## Gravity Operations
- gravity_left: Move all colored (non-zero) cells to the left within their row while preserving order
- gravity_right: Move all colored (non-zero) cells to the right within their row while preserving order
- gravity_up: Move all colored (non-zero) cells upward within their column while preserving order
- gravity_down: Move all colored (non-zero) cells downward within their column while preserving order

## Repetition & Mirroring
- repeatX: Repeat the grid horizontally (side by side)
- repeatY: Repeat the grid vertically (stacked)
- mirrorX: Append a horizontal reflection of the grid
- mirrorY: Append a vertical reflection of the grid

# Valid channels: analysis, commentary, final"""
            },
            {
                "role": "user",
                "content": f"""# ARC Puzzle Solver - Training Data Generation

You are solving Abstract Reasoning Corpus (ARC) puzzles to create training examples.

For each puzzle:
1. Use the analysis channel for examining patterns and reasoning
2. Identify the transformation rule from training examples
3. Apply the rule to the test input
4. Switch to the final channel for DETAILED STEP-BY-STEP REASONING

You MUST use both channels: first analysis, then final.

## Example of correct format:

{incontext_example}

## Task
Solve this ARC puzzle:

{examples_text}

Test Input:
{self.grid_to_string(test_input)}

GIVEN INFORMATION FOR GUIDANCE:
- Expected Output:
{expected_output}

- Transformation Hint (use this to guide your analysis, but explain the pattern in your own words):
{transformation_hint}

CRITICAL INSTRUCTIONS:
- ALWAYS use <|channel|>analysis first to work through the pattern
- ALWAYS switch to <|channel|>final for DETAILED, VERBOSE step-by-step reasoning
- In the final channel, provide EXTENSIVE reasoning with multiple steps explaining the transformation
- DO NOT output the resulting grid in the final channel - ONLY provide reasoning
- DO NOT mention technical terms like "gravity_left", "DSL", "program", or any code-like names
- Explain the transformation by analyzing the visual/logical patterns directly
- Focus on what changes between input and output (positions, colors, arrangements)
- Provide detailed reasoning that thoroughly explains each aspect of the transformation
- Your explanation should be self-contained and based purely on observable transformations
- Write at least 6-10 detailed reasoning steps in the final channel

Generate detailed step-by-step reasoning (NO GRID OUTPUT) in the final channel."""
            }
        ]

        return messages

    def analyze_arc_data(self, helmholtz_data: Dict, program: str = None, expected_output: str = None, max_retries: int = 5) -> Dict[str, str]:
        """
        Analyze ARC puzzle data and return structured analysis
        REQUIRES final channel - retries up to max_retries times

        Args:
            helmholtz_data: Helmholtz generated data
            program: Optional program information
            expected_output: Optional expected output for guided analysis
            max_retries: Maximum retry attempts (default: 5)

        Returns:
            Dictionary with analysis results, or None if all retries fail
        """
        # Create guided prompt once (reuse for retries)
        messages = self.create_guided_prompt(helmholtz_data, program or "unknown", expected_output or "unknown")

        attempt = 0
        current_reasoning_effort = "medium"  # Start with medium, fallback to low after 3 failures
        while attempt < max_retries:
            attempt += 1
            try:
                self.logger.info(f"Analyzing ARC data with GPT-OSS (Attempt {attempt})...")

                # Apply chat template (following sequential_validation_v4.py)
                self.logger.info("Applying chat template...")
                try:
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        reasoning_effort=current_reasoning_effort
                    ).to(self.model.device)
                    self.logger.info(f"Using reasoning_effort={current_reasoning_effort}")
                except TypeError:
                    # Fallback without reasoning_effort
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True
                    ).to(self.model.device)
                    self.logger.info("reasoning_effort not supported, using default")

                self.logger.info(f"Input tokens: {inputs['input_ids'].shape[1]:,}")
                self.logger.info("Starting GPT-OSS inference...")

                start_time = time.time()

                # Generate response (following sequential_validation_v4.py parameters)
                with torch.no_grad():
                    # Derive EOS token IDs
                    eos_ids = []
                    if self.tokenizer.eos_token_id is not None:
                        eos_ids.append(int(self.tokenizer.eos_token_id))
                    try:
                        ret_id = self.tokenizer.convert_tokens_to_ids("<|return|>")
                        if isinstance(ret_id, int) and ret_id >= 0:
                            eos_ids.append(ret_id)
                    except Exception:
                        pass

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10000,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=eos_ids if eos_ids else self.tokenizer.eos_token_id,
                    )

                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=False
                )

                inference_time = time.time() - start_time
                self.logger.info(f"GPT-OSS inference completed in {inference_time:.1f}s")
                self.logger.info(f"Response length: {len(response):,} characters")

                # Clean up tensors
                del outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Parse response using channel structure
                parsed_analysis = self._parse_gpt_oss_response(response)

                # CHECK: Does response have final channel?
                if parsed_analysis.get('channel_final_used', False):
                    # SUCCESS: Final channel present
                    self.logger.info(f"✅ Final channel found after {attempt} attempt(s) - analysis complete")

                    # Add metadata
                    analysis_result = {
                        'model_name': 'GPT-OSS-20B',
                        'raw_response': response,
                        'inference_time': inference_time,
                        'attempts_needed': attempt,
                        **parsed_analysis
                    }

                    return analysis_result
                else:
                    # RETRY: Final channel missing
                    if attempt < max_retries:
                        self.logger.warning(f"⚠️  Final channel missing (Attempt {attempt}/{max_retries}) - retrying...")

                        # After 3 failures, switch to reasoning_effort="low"
                        if attempt >= 3 and current_reasoning_effort == "medium":
                            self.logger.warning(f"⚠️  Switching to reasoning_effort='low' after {attempt} failed attempts")
                            current_reasoning_effort = "low"

                        continue
                    else:
                        self.logger.error(f"❌ Failed to get final channel after {max_retries} attempts")
                        return None

            except Exception as e:
                self.logger.error(f"Failed to analyze with GPT-OSS (Attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    self.logger.info("Retrying due to exception...")
                    continue
                else:
                    self.logger.error(f"❌ All {max_retries} attempts failed")
                    return None

        # All retries exhausted
        self.logger.error(f"❌ Failed after {max_retries} attempts")
        return None

    def _parse_gpt_oss_response(self, response: str) -> Dict[str, str]:
        """Parse GPT-OSS response using channel structure"""
        parsed = {}

        try:
            # Check for channel usage
            has_analysis = "<|channel|>analysis" in response
            has_final = "<|channel|>final" in response

            if has_analysis:
                # Extract analysis content
                analysis_start = response.find("<|channel|>analysis<|message|>")
                if analysis_start != -1:
                    analysis_content = response[analysis_start + len("<|channel|>analysis<|message|>"):]

                    # Find end of analysis (either final channel or end)
                    final_start = analysis_content.find("<|channel|>final")
                    if final_start != -1:
                        analysis_content = analysis_content[:final_start]

                    # Try to parse structured content or use full content
                    structured = parse_step_response(analysis_content)
                    if structured:
                        parsed.update(structured)
                    else:
                        parsed['pattern_recognition'] = analysis_content.strip()
                        parsed['transformation_rule'] = "See pattern recognition"
                        parsed['verification_process'] = "See analysis content"

            if has_final:
                # Extract final grid
                final_start = response.find("<|channel|>final<|message|>")
                if final_start != -1:
                    final_content = response[final_start + len("<|channel|>final<|message|>"):]

                    # Clean up final content
                    end_markers = ["<|return|>", "<|end|>"]
                    for marker in end_markers:
                        if marker in final_content:
                            final_content = final_content[:final_content.find(marker)]

                    parsed['final_grid'] = final_content.strip()
                    # Also store final content separately for embeddings
                    parsed['full_response'] = final_content.strip()

            # Channel usage metadata
            parsed['channel_analysis_used'] = has_analysis
            parsed['channel_final_used'] = has_final

        except Exception as e:
            self.logger.warning(f"Failed to parse GPT-OSS response structure: {e}")
            # Fallback to basic parsing
            parsed['pattern_recognition'] = response[:500] if response else "No response"
            parsed['transformation_rule'] = "Could not parse transformation rule"
            parsed['verification_process'] = "Could not parse verification"

        return parsed

    def extract_final_channel_only(self, response: str) -> str:
        """
        Extract ONLY final channel text for embeddings
        This is shorter and more focused than full analysis text

        Args:
            response: Full GPT-OSS response

        Returns:
            Final channel text only, or empty string if not found
        """
        try:
            final_start = response.find("<|channel|>final<|message|>")
            if final_start == -1:
                return ""

            final_content = response[final_start + len("<|channel|>final<|message|>"):]

            # Clean up final content
            end_markers = ["<|return|>", "<|end|>", "<|channel|>"]
            for marker in end_markers:
                if marker in final_content:
                    final_content = final_content[:final_content.find(marker)]

            return final_content.strip()

        except Exception as e:
            self.logger.warning(f"Failed to extract final channel: {e}")
            return ""

    def analyze_multiple_samples(self, samples_data: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze multiple Helmholtz samples

        Args:
            samples_data: List of Helmholtz sample data

        Returns:
            Dictionary mapping sample IDs to analysis results
        """
        results = {}

        for i, sample_data in enumerate(samples_data, 1):
            self.logger.info(f"Processing sample {i}/{len(samples_data)}")

            try:
                # Extract program and expected output
                program = sample_data.get('program', 'unknown')
                expected_output = self._format_expected_output(sample_data)

                analysis = self.analyze_arc_data(
                    sample_data,
                    program=program,
                    expected_output=expected_output
                )

                results[f"sample_{i}"] = {
                    'task_id': sample_data.get('task_id', f'sample_{i}'),
                    'analysis': analysis,
                    'metadata': {
                        'program': program,
                        'analysis_source': 'gpt_oss'
                    }
                }

            except Exception as e:
                self.logger.error(f"Failed to process sample {i}: {e}")
                results[f"sample_{i}"] = {
                    'error': str(e),
                    'task_id': sample_data.get('task_id', f'sample_{i}')
                }

        self.logger.info(f"Completed analysis of {len(samples_data)} samples")
        return results

    def _format_expected_output(self, sample_data: Dict) -> str:
        """Format expected output as string"""
        try:
            test_output = sample_data['test'][0]['output']
            output_grid = np.array(test_output)
            return self.grid_to_string(output_grid)
        except Exception:
            return "unknown"

    def cleanup(self):
        """Clean up model resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("GPT-OSS model resources cleaned up")