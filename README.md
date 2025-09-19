# torch-code-tracing
Code trace your model

```py
from torch_code_tracing import TracingMode

with TracingMode():
    out = model(*args, **kwargs)
```
