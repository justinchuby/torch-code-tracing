from __future__ import annotations

__all__ = ["TracingMode"]

from collections.abc import Sequence
import dataclasses
import inspect

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

_GRAY = "\033[2m"
_RESET = "\033[0m"


DTYPE_ABBRS = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.float8_e4m3fn: "f8e4m3fn",
    torch.float8_e5m2: "f8e5m2",
    torch.float8_e4m3fnuz: "f8e4m3fnuz",
    torch.float8_e5m2fnuz: "f8e5m2fnuz",
    torch.float8_e8m0fnu: "f8e8m0fnu",
    # torch.float4_e2m1fn_x2: "f4e2m1fnx2",
    torch.complex32: "c32",
    torch.complex64: "c64",
    torch.complex128: "c128",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
    torch.uint16: "u16",
    torch.uint32: "u32",
    torch.uint64: "u64",
    torch.bits16: "b16",
    torch.bits1x8: "b1x8",
}


def _stringify_shape(shape) -> str:
    return f"[{', '.join([str(x) for x in shape])}]"


def _tensor_debug_string(tensor) -> str:
    """Convert tensor to debug string representation."""
    if isinstance(tensor, torch.Tensor):
        return f"{DTYPE_ABBRS[tensor.dtype]}{_stringify_shape(tensor.shape)}"
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
    def __init__(self, *args, quiet: bool = False, **kwargs):
        """A TorchDispatchMode that prints code traces of all tensor operations.

        Args:
            quiet: If True, only store the traces and do not print them immediately.
        """
        super().__init__(*args, **kwargs)
        self.traces: list[Trace] = []
        self._verbose = not quiet

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        stack = reversed(inspect.stack()[1:])  # Exclude the current frame
        # Filter out frames from PyTorch internals
        stack = [
            frame for frame in stack if "site-packages/torch" not in frame.filename
        ]
        op_str = _op_to_str(func, *args, **kwargs)

        result = func(*args, **kwargs)

        output_str = _arg_to_str(result)
        if isinstance(result, Sequence):
            output_str = f"({output_str})"

        self._add_trace(Trace(f"{op_str} -> {output_str}", stack))

        return result

    def print(self) -> None:
        """Print the formatted trace to stdout."""
        for i in range(len(self.traces)):
            self._print_trace(i)

    def format(self, color: bool = False) -> str:
        """Return the formatted trace as a string."""
        lines = []
        for i in range(len(self.traces)):
            lines.append(self._trace_str(i, color=color))
        return "\n".join(lines)

    def _add_trace(self, trace: Trace) -> None:
        self.traces.append(trace)
        if self._verbose:
            self._print_trace(-1)

    def _print_trace(self, index: int) -> None:
        trace_str = self._trace_str(index, color=True)
        # Apply color formatting to the comment portion (after #)
        formatted_str = (
            trace_str.replace("  # ", f"  {_GRAY}# ").replace("\n", f"{_RESET}\n")
            + _RESET
        )
        print(formatted_str)

    def _trace_str(self, index: int, color: bool = False) -> str:
        if not self.traces:
            return "<no traces>"

        trace = self.traces[index]

        common_length = 0

        if index < 0:
            index = index + len(self.traces)

        if index > 0:
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

            src_line = frame.code_context[0] if frame.code_context else ""

            if color and src_line:
                if (positions := frame.positions) is not None:
                    if (
                        positions.lineno == positions.end_lineno == frame.lineno
                        and positions.col_offset is not None
                        and positions.end_col_offset is not None
                    ):
                        # Highlight the exact column if available
                        start_col = positions.col_offset
                        end_col = positions.end_col_offset
                        src_line = (
                            src_line[:start_col]
                            + "\033[36m"  # Cyan text
                            + src_line[start_col:end_col]
                            + _RESET
                            + src_line[end_col:]
                        )

            src_line = src_line.strip()

            if i == len(relevant_stack) - 1:
                # Last frame. Show the operator call
                op_str = f"{trace.op_str};"
            else:
                op_str = "⬇️"

            if color:
                line = f"{'│ ' * indent}{src_line}  {_GRAY}# {frame.filename}:{frame.lineno} in {frame.function}: {op_str}{_RESET}"
            else:
                line = f"{'│ ' * indent}{src_line}  # {frame.filename}:{frame.lineno} in {frame.function}: {op_str}"
            lines.append(line)

        return "\n".join(lines)
