# torch-code-tracing

Code trace your PyTorch model

```py
from torch_code_tracing import TracingMode

with TracingMode():
    out = model(*args, **kwargs)
```

```py
out = model(**example_kwargs)  # /home/justinchu/dev/onnxscript/test.py:41 in <module>: ⬇️
| output = func(self, *args, **kwargs)  # /home/justinchu/anaconda3/envs/onnx/lib/python3.13/site-packages/transformers/utils/generic.py:969 in wrapper: ⬇️
| | inputs_embeds = self.get_input_embedding [...]  # /home/justinchu/anaconda3/envs/onnx/lib/python3.13/site-packages/transformers/models/gemma3/modeling_gemma3.py:1175 in forward: ⬇️
| | | return super().forward(input_ids) * self [...]  # /home/justinchu/anaconda3/envs/onnx/lib/python3.13/site-packages/transformers/models/gemma3/modeling_gemma3.py:144 in forward: embedding(bf16[262208, 2560], i64[2, 3], 0);
out = model(**example_kwargs)  # /home/justinchu/dev/onnxscript/test.py:41 in <module>: ⬇️
| output = func(self, *args, **kwargs)  # /home/justinchu/anaconda3/envs/onnx/lib/python3.13/site-packages/transformers/utils/generic.py:969 in wrapper: ⬇️
| | inputs_embeds = self.get_input_embedding [...]  # /home/justinchu/anaconda3/envs/onnx/lib/python3.13/site-packages/transformers/models/gemma3/modeling_gemma3.py:1175 in forward: ⬇️
...
```
