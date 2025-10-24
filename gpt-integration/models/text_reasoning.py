"""
Text Reasoning Module using LLaMA-8B
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional


class TextReasoningModule(nn.Module):
    """
    LLaMA-8B wrapper for text reasoning

    Generates reasoning text and extracts z_init for grid generation
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-8B-Instruct",
        freeze: bool = True,
        device: str = "cuda:3"
    ):
        super().__init__()

        # When CUDA_VISIBLE_DEVICES is set, use "auto" or "cuda" instead of specific device number
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.freeze = freeze

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Auto device mapping handles CUDA_VISIBLE_DEVICES correctly
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            print("✓ LLaMA-8B frozen")

        self.hidden_size = self.model.config.hidden_size  # 4096 for LLaMA-8B

    def _prepare_prompt(
        self,
        problem_text: str,
        latent_prefix: Optional[torch.Tensor] = None,
        max_length: int = 2048
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Tokenize a single prompt and optionally append a latent prefix token."""

        prompt = problem_text + "\n\nLet me reason step by step before predicting the grid."
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        if latent_prefix is None:
            return inputs, None, None

        prefix = latent_prefix.to(self.device)
        prefix = prefix.to(self.model.dtype)
        prefix = prefix.view(1, 1, -1)

        embed_tokens = self.model.model.embed_tokens
        token_embeds = embed_tokens(inputs["input_ids"]).to(self.model.dtype)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

        attention_mask = torch.cat(
            [torch.ones((inputs["attention_mask"].shape[0], 1), device=self.device), inputs["attention_mask"]],
            dim=1
        )

        return inputs, inputs_embeds, attention_mask

    def generate_latent(
        self,
        problem_text: str,
        latent_prefix: Optional[torch.Tensor] = None,
        max_length: int = 128,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, str]:
        """Generate reasoning for a single example and return final hidden state + text."""

        inputs, inputs_embeds, attention_mask = self._prepare_prompt(problem_text, latent_prefix)

        gen_kwargs = dict(
            max_new_tokens=max_length,
            do_sample=False,  # Greedy decoding for deterministic generation
            return_dict_in_generate=True,
            output_hidden_states=True
        )

        with torch.no_grad():
            if inputs_embeds is None:
                outputs = self.model.generate(**inputs, **gen_kwargs)
            else:
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )

        sequences = outputs.sequences
        generated_text = self.tokenizer.decode(sequences[0], skip_special_tokens=True)

        last_step_hidden = outputs.hidden_states[-1]
        last_layer_hidden = last_step_hidden[-1]
        z_init = last_layer_hidden[:, -1, :].squeeze(0).to(torch.float32)

        return z_init, generated_text

    def forward(
        self,
        problem_texts: List[str],
        latent_prefixes: Optional[List[Optional[torch.Tensor]]] = None,
        max_length: int = 128,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Sequentially run generation for each problem (batches are tiny)."""

        if latent_prefixes is None:
            latent_prefixes = [None] * len(problem_texts)

        z_list = []
        texts = []
        for prompt, latent_prefix in zip(problem_texts, latent_prefixes):
            z_init, generated = self.generate_latent(
                prompt,
                latent_prefix=latent_prefix,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
            )
            z_list.append(z_init)
            texts.append(generated)

        return torch.stack(z_list, dim=0), texts

    def generate_text_only(
        self,
        problem_texts: List[str],
        latent_prefixes: Optional[List[Optional[torch.Tensor]]] = None,
        max_length: int = 128,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> List[str]:
        """Return reasoning texts for analysis/debugging."""

        if latent_prefixes is None:
            latent_prefixes = [None] * len(problem_texts)

        texts = []
        for prompt, latent_prefix in zip(problem_texts, latent_prefixes):
            _, generated = self.generate_latent(
                prompt,
                latent_prefix=latent_prefix,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
            )
            texts.append(generated)

        return texts


# Test
if __name__ == "__main__":
    print("Testing TextReasoningModule...")

    module = TextReasoningModule(
        model_name="meta-llama/Llama-3.2-8B-Instruct",
        freeze=True,
        device="cuda:3"
    )

    problem_texts = [
        "Solve this ARC puzzle: The pattern is to rotate 90 degrees clockwise."
    ]

    z_init, _ = module(problem_texts, max_length=50)
    print(f"z_init shape: {z_init.shape}")  # Should be [1, 4096]

    print("✅ TextReasoningModule test passed!")
