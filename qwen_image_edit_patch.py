import torch
import diffusers.models.transformers.transformer_qwenimage as qwen_image_module
from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope

def apply_qwen_image_patches():
    # Patch 1: if newer diffusers doesn't pass txt_seq_lens, use a safe max
    original_qwen_rope_forward = QwenEmbedRope.forward

    def patched_qwen_rope_forward(self, video_fhw, txt_seq_lens=None, device=None, max_txt_seq_len=None):
        # Newer diffusers can call this with both args missing.
        # Use a safe upper bound and let the rotary application slice later.
        if max_txt_seq_len is None and txt_seq_lens is None:
            max_txt_seq_len = 4096
        return original_qwen_rope_forward(
            self,
            video_fhw,
            txt_seq_lens,
            device,
            max_txt_seq_len,
        )

    QwenEmbedRope.forward = patched_qwen_rope_forward

    # Patch 2: slice oversized rotary frequencies to the actual seq length
    original_apply_rotary_emb_qwen = qwen_image_module.apply_rotary_emb_qwen

    def patched_apply_rotary_emb_qwen(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
        seq_len = x.shape[1]

        if not use_real and isinstance(freqs_cis, torch.Tensor):
            if freqs_cis.dim() == 3 and freqs_cis.shape[1] > seq_len:
                freqs_cis = freqs_cis[:, :seq_len]
            elif freqs_cis.dim() == 2 and freqs_cis.shape[0] > seq_len:
                freqs_cis = freqs_cis[:seq_len]

        return original_apply_rotary_emb_qwen(
            x, freqs_cis, use_real, use_real_unbind_dim
        )

    qwen_image_module.apply_rotary_emb_qwen = patched_apply_rotary_emb_qwen

apply_qwen_image_patches()
