from __future__ import annotations

__all__ = ["TracingMode"]

import dataclasses
import inspect

import torch
from torch.utils._dtype_abbrs import dtype_abbrs
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


_GRAY = "\033[2m"
_RESET = "\033[0m"


def _stringify_shape(shape) -> str:
    return f"[{', '.join([str(x) for x in shape])}]"


def _tensor_debug_string(tensor) -> str:
    """Convert tensor to debug string representation."""
    if isinstance(tensor, torch.Tensor):
        return f"{dtype_abbrs[tensor.dtype]}{_stringify_shape(tensor.shape)}"
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def _arg_to_str(arg) -> str:
    def to_str(x):
        if isinstance(x, torch.Tensor):
            return _tensor_debug_string(x)
        return x

    arg = tree_map(to_str, arg)
    return str(arg)


def _op_to_str(op, *args, **kwargs) -> str:
    args_str = ", ".join(_arg_to_str(arg) for arg in args)

    if kwargs:
        kwargs_str = ", " + ", ".join(
            f"{k}={_arg_to_str(v)}" for k, v in kwargs.items()
        )
    else:
        kwargs_str = ""

    if isinstance(op, torch._ops.OpOverload):
        op_name = op.__qualname__
    elif hasattr(op, "__module__") and hasattr(op, "__name__"):
        op_name = f"{op.__module__}.{op.__name__}"
    else:
        op_name = str(op)

    if op_name.startswith("aten::"):
        op_name = op_name[len("aten::") :]

    return f"{op_name}({args_str}{kwargs_str})"


@dataclasses.dataclass
class Trace:
    op_str: str
    # Outer most frame is the first element. This is a reversed of inspect.stack()
    stack: list[inspect.FrameInfo]


class TracingMode(TorchDispatchMode):
    def __init__(self, *args, verbose: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.traces: list[Trace] = []
        self._verbose = verbose

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        stack = reversed(inspect.stack()[1:])  # Exclude the current frame
        # Filter out frames from PyTorch internals
        stack = [
            frame for frame in stack if "site-packages/torch" not in frame.filename
        ]
        op_str = _op_to_str(func, *args, **kwargs)
        self._add_trace(Trace(op_str, stack))

        result = func(*args, **kwargs)

        return result

    def print(self) -> None:
        for i in range(len(self.traces)):
            self._print_trace(i)

    def _add_trace(self, trace: Trace) -> None:
        self.traces.append(trace)
        if self._verbose:
            self._print_trace(-1)

    def _print_trace(self, index: int) -> None:
        trace_str = self._trace_str(index)
        # Apply color formatting to the comment portion (after #)
        formatted_str = (
            trace_str.replace("  # ", f"  {_GRAY}# ").replace("\n", f"{_RESET}\n")
            + _RESET
        )
        print(formatted_str)

    def _trace_str(self, index: int) -> str:
        if not self.traces:
            return "<no traces>"

        if index < 0:
            index = len(self.traces) + index
        if index >= len(self.traces):
            raise IndexError("Trace index out of range")

        trace = self.traces[index]

        common_length = 0

        if index > 1:
            # Find the common prefix between the current stack and the trace stack
            prev_trace = self.traces[index - 1]
            for f1, f2 in zip(trace.stack, prev_trace.stack):
                if f1.filename == f2.filename and f1.lineno == f2.lineno:
                    common_length += 1
                else:
                    break
            if common_length == len(trace.stack):
                # Keep at least one frame to show the context of the operator
                common_length -= 1
            relevant_stack = trace.stack[common_length:]
        else:
            relevant_stack = trace.stack

        lines = []
        for i, frame in enumerate(relevant_stack):
            indent = i + common_length
            src_line = frame.code_context[0].strip() if frame.code_context else ""
            if len(src_line) > 40:
                src_line = f"{src_line[:40]} [...]"

            if i == len(relevant_stack) - 1:
                # Last frame. Show the operator call
                op_str = f"{trace.op_str};"
            else:
                op_str = "⬇️"

            lines.append(
                f"{'| ' * indent}{src_line}  # {frame.filename}:{frame.lineno} in {frame.function}: {op_str}"
            )

        return "\n".join(lines)
