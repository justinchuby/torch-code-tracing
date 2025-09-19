# torch-code-tracing

Code trace your PyTorch model

## Install

```sh
pip install git+https://github.com/justinchuby/torch-code-tracing.git
```

## Usage

```py
from torch_code_tracing import TracingMode

with TracingMode():
    out = model(*args, **kwargs)
```

```py
out = model(**example_kwargs)  # test.py:41 in <module>:
│ ⬇️
│ output = func(self, *args, **kwargs)  # site-packages/transformers/utils/generic.py:969 in wrapper:
│ │ ⬇️
│ │ inputs_embeds = self.get_input_embeddings()(llm_input_ids)  # site-packages/transformers/models/gemma3/modeling_gemma3.py:1175 in forward:
│ │ │ ⬇️
│ │ │ return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)  # site-packages/transformers/models/gemma3/modeling_gemma3.py:144 in forward:
│ │ │ │ # embedding(bf16[262208, 2560], i64[2, 3], 0) -> bf16[2, 3, 2560];
│ │ │ return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)  # site-packages/transformers/models/gemma3/modeling_gemma3.py:144 in forward:
│ │ │ │ # mul.Tensor(bf16[2, 3, 2560], bf16[]) -> bf16[2, 3, 2560];
│ │ cache_position = torch.arange(  # site-packages/transformers/models/gemma3/modeling_gemma3.py:1179 in forward:
│ │ │ # arange.start(30, 33, device=meta, pin_memory=False) -> i64[3];
│ │ causal_mask = self._update_causal_mask(  # site-packages/transformers/models/gemma3/modeling_gemma3.py:1205 in forward:
│ │ │ ⬇️
│ │ │ causal_mask = torch.full(  # site-packages/transformers/models/gemma3/modeling_gemma3.py:1050 in _update_causal_mask:
│ │ │ │ # full([3, 33], -3.3895313892515355e+38, dtype=torch.bfloat16, device=meta, pin_memory=False) -> bf16[3, 33];
│ │ │ causal_mask = torch.triu(causal_mask, diagonal=1)  # site-packages/transformers/models/gemma3/modeling_gemma3.py:1056 in _update_causal_mask:
│ │ │ │ # triu(bf16[3, 33], 1) -> bf16[3, 33];
...
```
