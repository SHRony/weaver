from pathlib import Path
from typing import Any

import torch


def main() -> None:
    path = Path("~/.cache/weaver/gpt2.pt").expanduser()

    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained("gpt2")  # pyright: ignore[reportUnknownMemberType]
    # walk hf.state_dict(): skip ("attn","bias") by segment match, transpose the
    # 4 Conv1D suffixes, copy_ into model.state_dict()[key] under torch.no_grad()
    hf_dict = hf.state_dict()
    model_dict : dict[str, Any] = {}
    transposed_keys = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]
    for key in hf_dict:
        if key.split(".")[-2:] == ["attn", "bias"]:
            continue
        if any(key.endswith(k) for k in transposed_keys):
            model_dict[key] = hf_dict[key].transpose(0, 1).contiguous()
        else:
            model_dict[key] = hf_dict[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_dict, path)
    print(f"saved {len(model_dict)} tensors to {path}")
if __name__ == "__main__":
    main()