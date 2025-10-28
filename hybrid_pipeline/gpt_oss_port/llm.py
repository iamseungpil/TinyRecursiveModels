"""
Text Reasoning Module wrapping GPT-OSS (or other HF causal LLMs).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LatentOutput:
    """Container for LLM latent representations returned by TextReasoningModule."""

    final_hidden: torch.Tensor
    generated_text: str
    hidden_sequence: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None


class TextReasoningModule(nn.Module):
    """GPT-OSS wrapper for text reasoning latents."""

    def __init__(
        self,
        model_name: str = "unsloth/gpt-oss-mxfp4-20b",
        freeze: bool = True,
        device: str = "cuda",
        extract_full_sequence: bool = False,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        device_map: Optional[str] = "auto",
        trust_remote_code: bool = True,
        use_fast_tokenizer: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            freeze: If True, freeze all model parameters
            device: Device to load model on
        """
        super().__init__()

        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze = freeze
        self.model_name = model_name
        self.extract_full_sequence = extract_full_sequence

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_tokenizer,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            print("✓ LLaMA frozen")

        self.hidden_size = self.model.config.hidden_size

    def _prepare_prompt(
        self,
        problem_text: str,
        latent_prefix: Optional[torch.Tensor] = None,
        max_length: int = 2048,
        use_chat_template: bool = False,
        reasoning_effort: Optional[str] = None,
    ) -> Tuple[dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Tokenize prompt and optionally prepend latent prefix.

        Note: Does NOT add "Let me reason step by step" here.
        That should be added by format_problem_text() in adapters/feedback_formatter.py

        Args:
            problem_text: Input prompt string
            latent_prefix: Optional [hidden_size] tensor to prepend
            max_length: Maximum sequence length
            use_chat_template: Use apply_chat_template (for GPT-OSS compatibility)
            reasoning_effort: Optional reasoning effort level ("low", "medium", "high")
                            Only works if model's chat template supports it (e.g., OpenAI o1)

        Returns:
            inputs: Tokenized inputs
            inputs_embeds: Embeddings with latent prefix (if provided)
            attention_mask: Attention mask with latent prefix (if provided)
        """
        # Use chat template if requested (e.g., for GPT-OSS)
        if use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": problem_text}
            ]

            # Try with reasoning_effort first (OpenAI o1 style)
            if reasoning_effort is not None:
                try:
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        truncation=True,
                        max_length=max_length,
                        reasoning_effort=reasoning_effort  # Model-specific parameter
                    ).to(self.device)
                except TypeError:
                    # Fallback: model doesn't support reasoning_effort
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        truncation=True,
                        max_length=max_length
                    ).to(self.device)
            else:
                # No reasoning_effort requested
                try:
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        truncation=True,
                        max_length=max_length
                    ).to(self.device)
                except Exception as e:
                    # Fallback to plain tokenization if chat template fails
                    print(f"Warning: apply_chat_template failed ({e}), using plain tokenization")
                    inputs = self.tokenizer(
                        problem_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    ).to(self.device)
        else:
            # Plain tokenization (LLaMA default)
            inputs = self.tokenizer(
                problem_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

        if latent_prefix is None:
            return inputs, None, None

        # Prepend latent prefix to token embeddings
        prefix = latent_prefix.to(self.device)
        prefix = prefix.to(self.model.dtype)
        prefix = prefix.view(1, 1, -1)  # [1, 1, hidden_size]

        embed_tokens = self.model.model.embed_tokens
        token_embeds = embed_tokens(inputs["input_ids"]).to(self.model.dtype)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

        attention_mask = torch.cat(
            [torch.ones((inputs["attention_mask"].shape[0], 1), device=self.device),
             inputs["attention_mask"]],
            dim=1
        )

        return inputs, inputs_embeds, attention_mask

    def _extract_hidden_sequence(
        self,
        sequences: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-run the base model to obtain full hidden states for the generated sequence."""

        sequences = sequences.to(self.device)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", self.tokenizer.eos_token_id)
        attention_mask = sequences.ne(pad_token_id).to(self.device)

        with torch.no_grad():
            hidden_outputs = self.model.model(
                sequences,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        hidden_sequence = hidden_outputs.hidden_states[-1].to(torch.float32)
        return hidden_sequence, attention_mask.bool()

    def _run_generation(
        self,
        inputs: dict,
        gen_kwargs: dict,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        need_full_sequence: bool = False,
    ) -> LatentOutput:
        """Run HuggingFace generation and package the latent outputs."""

        with torch.no_grad():
            if inputs_embeds is None:
                outputs = self.model.generate(**inputs, **gen_kwargs)
            else:
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

        sequences = outputs.sequences
        generated_text = self.tokenizer.decode(sequences[0], skip_special_tokens=True)

        # Final hidden state from last generated token
        last_step_hidden = outputs.hidden_states[-1]
        last_layer_hidden = last_step_hidden[-1]
        final_hidden = last_layer_hidden[:, -1, :].squeeze(0).to(torch.float32)

        hidden_sequence = None
        full_attention_mask = None
        if need_full_sequence:
            hidden_sequence, full_attention_mask = self._extract_hidden_sequence(sequences)

        return LatentOutput(
            final_hidden=final_hidden,
            generated_text=generated_text,
            hidden_sequence=hidden_sequence.squeeze(0) if hidden_sequence is not None else None,
            attention_mask=full_attention_mask.squeeze(0) if full_attention_mask is not None else None,
        )

    def generate_latent(
        self,
        problem_text: str,
        latent_prefix: Optional[torch.Tensor] = None,
        max_length: int = 128,
        use_chat_template: bool = False,
        reasoning_effort: Optional[str] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        extract_full_sequence: Optional[bool] = None,
    ) -> LatentOutput:
        """
        Generate reasoning text and associated latent representations.

        Args:
            problem_text: Input prompt
            latent_prefix: Optional latent prefix to prepend
            max_length: Maximum new tokens to generate
            use_chat_template: Use apply_chat_template for GPT-OSS compatibility
            reasoning_effort: Optional reasoning effort level ("low", "medium", "high")
                            Only effective if model's chat template supports it
            do_sample: Whether to use sampling (vs greedy)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            extract_full_sequence: Extract full hidden sequence (for cross-attention)

        Returns:
            LatentOutput: Container with final_hidden, generated_text, and optional sequence
        """
        prompt_max_length = getattr(self.model.config, "max_position_embeddings", 2048)

        need_full_sequence = (
            self.extract_full_sequence if extract_full_sequence is None else extract_full_sequence
        )

        inputs, inputs_embeds, attention_mask = self._prepare_prompt(
            problem_text,
            latent_prefix=latent_prefix,
            max_length=prompt_max_length,
            use_chat_template=use_chat_template,
            reasoning_effort=reasoning_effort,
        )

        # Generation parameters (following gpt_oss_analyzer.py reference)
        gen_kwargs = dict(
            max_new_tokens=max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return self._run_generation(
            inputs=inputs,
            gen_kwargs=gen_kwargs,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            need_full_sequence=need_full_sequence,
        )

    def generate_with_repulsion(
        self,
        problem_description: str,
        failed_latent: torch.Tensor,
        repulsion_weight: float = 0.5,
        max_length: int = 128
    ) -> LatentOutput:
        """
        Generate with latent repulsion AWAY from failed_latent.

        Push generation away from failed/incorrect latent representations.

        Args:
            problem_description: Input prompt
            failed_latent: [hidden_size] latent to AVOID (repel from)
            repulsion_weight: How strongly to push away (0.0 = no effect, 1.0 = max)
            max_length: Maximum new tokens

        Returns:
            z_init: [hidden_size] final hidden state
            generated_text: Generated reasoning text
        """
        latent = self.generate_latent(
            problem_description,
            latent_prefix=None,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            extract_full_sequence=self.extract_full_sequence,
        )

        direction = latent.final_hidden - failed_latent.to(latent.final_hidden.device)
        direction = direction / (direction.norm() + 1e-8)
        repelled = latent.final_hidden + repulsion_weight * 0.1 * direction

        return LatentOutput(
            final_hidden=repelled,
            generated_text=latent.generated_text,
            hidden_sequence=latent.hidden_sequence,
            attention_mask=latent.attention_mask,
        )

    def forward(
        self,
        problem_texts: List[str],
        latent_prefixes: Optional[List[Optional[torch.Tensor]]] = None,
        max_length: int = 128,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Sequentially run generation for each problem.

        Args:
            problem_texts: List of input prompts
            latent_prefixes: Optional list of latent prefixes
            max_length: Maximum new tokens per generation

        Returns:
            z_list: [batch, hidden_size] stacked hidden states
            texts: List of generated text strings
        """
        if latent_prefixes is None:
            latent_prefixes = [None] * len(problem_texts)

        z_list = []
        texts = []
        for prompt, latent_prefix in zip(problem_texts, latent_prefixes):
            latent = self.generate_latent(
                prompt,
                latent_prefix=latent_prefix,
                max_length=max_length,
            )
            z_list.append(latent.final_hidden)
            texts.append(latent.generated_text)

        return torch.stack(z_list, dim=0), texts

    def generate_text_only(
        self,
        problem_texts: List[str],
        latent_prefixes: Optional[List[Optional[torch.Tensor]]] = None,
        max_length: int = 128,
    ) -> List[str]:
        """
        Return reasoning texts only (for analysis/debugging).

        Args:
            problem_texts: List of input prompts
            latent_prefixes: Optional list of latent prefixes
            max_length: Maximum new tokens per generation

        Returns:
            texts: List of generated text strings
        """
        if latent_prefixes is None:
            latent_prefixes = [None] * len(problem_texts)

        texts = []
        for prompt, latent_prefix in zip(problem_texts, latent_prefixes):
            latent = self.generate_latent(
                prompt,
                latent_prefix=latent_prefix,
                max_length=max_length,
            )
            texts.append(latent.generated_text)

        return texts


# Test
if __name__ == "__main__":
    print("Testing TextReasoningModule...")

    module = TextReasoningModule(
        model_name="meta-llama/Llama-3.2-8B-Instruct",
        freeze=True,
        device="cuda"
    )

    problem_texts = [
        "Solve this ARC puzzle: The pattern is to rotate 90 degrees clockwise."
    ]

    z_init, texts = module(problem_texts, max_length=50)
    print(f"z_init shape: {z_init.shape}")  # Should be [1, 4096]
    print(f"Generated text: {texts[0][:100]}...")

    print("✅ TextReasoningModule test passed!")
