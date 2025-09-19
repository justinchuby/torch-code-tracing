# torch-code-tracing

Code trace your PyTorch model

```py
from torch_code_tracing import TracingMode

with TracingMode():
    out = model(*args, **kwargs)
```

```py
out = model(**example_kwargs)  # test.py:41 in <module>: ⬇️
| output = func(self, *args, **kwargs)  # site-packages/transformers/utils/generic.py:969 in wrapper: ⬇️
| | inputs_embeds = self.get_input_embedding [...]  # site-packages/transformers/models/gemma3/modeling_gemma3.py:1175 in forward: ⬇️
| | | return super().forward(input_ids) * self [...]  # site-packages/transformers/models/gemma3/modeling_gemma3.py:144 in forward: embedding(bf16[262208, 2560], i64[2, 3], 0);
out = model(**example_kwargs)  # test.py:41 in <module>: ⬇️
| output = func(self, *args, **kwargs)  # site-packages/transformers/utils/generic.py:969 in wrapper: ⬇️
| | inputs_embeds = self.get_input_embedding [...]  # site-packages/transformers/models/gemma3/modeling_gemma3.py:1175 in forward: ⬇️
...
```
