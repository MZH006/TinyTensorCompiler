"""Export a small CPU float32 FX subset with fixed-shape runtime inputs.

Supported: rank-two tensor inputs/constants, positional matmul/add/relu,
and one tensor output. Example inputs validate shapes; their data is separate.
"""
import argparse
import operator
from pathlib import Path

import torch
from torch.fx import Node, symbolic_trace


def check_tensor(value):
    if not isinstance(value, torch.Tensor):
        raise ValueError("Expected a tensor")
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError("Only CPU float32 tensors are supported")
    if value.layout != torch.strided or value.ndim != 2 or min(value.shape) <= 0:
        raise ValueError("Only nonempty rank-two strided tensors are supported")
    if not torch.isfinite(value).all():
        raise ValueError("Tensor values must be finite")


def export_graph(model, inputs):
    """Return graph text; reject unsupported FX nodes before writing a file."""
    traced = symbolic_trace(model)
    env = {}
    lines = []
    placeholders = [node for node in traced.graph.nodes if node.op == "placeholder"]
    if len(inputs) != len(placeholders):
        raise ValueError("Input count does not match graph placeholders")
    supplied = iter(inputs)
    for node in traced.graph.nodes:
        try:
            if node.op in ("placeholder", "get_attr"):
                if node.op == "placeholder":
                    value = next(supplied)
                else:
                    value = traced
                    for part in node.target.split("."):
                        value = getattr(value, part)
                check_tensor(value)
                value = value.detach()
                env[node] = value
                if node.op == "placeholder":
                    lines.append(f"input {node.name} {value.shape[0]} {value.shape[1]}")
                else:
                    data = " ".join(map(repr, value.flatten().tolist()))
                    lines.append(f"tensor {node.name} {value.shape[0]} {value.shape[1]} {data}")
            elif node.op == "call_function" or (node.op == "call_method" and node.target == "add"):
                targets = {operator.matmul: "matmul", torch.matmul: "matmul",
                           operator.add: "add", torch.add: "add", torch.relu: "relu"}
                opcode = "add" if node.op == "call_method" else targets.get(node.target)
                if opcode is None or node.kwargs:
                    raise ValueError("Unsupported function or keyword arguments")
                arity = 1 if opcode == "relu" else 2
                if len(node.args) != arity or any(not isinstance(arg, Node) for arg in node.args):
                    raise ValueError("Expected positional tensor operands")
                args = [env[arg] for arg in node.args]
                if opcode == "add" and args[0].shape != args[1].shape:
                    raise ValueError("Broadcasting is not supported")
                if opcode == "matmul" and args[0].shape[1] != args[1].shape[0]:
                    raise ValueError("MatMul inner dimensions must match")
                with torch.no_grad():
                    function = {"matmul": torch.matmul, "add": torch.add, "relu": torch.relu}[opcode]
                    value = function(*args)
                check_tensor(value)
                env[node] = value
                lines.append(f"{opcode} {node.name} " + " ".join(arg.name for arg in node.args))
            elif node.op == "output":
                if len(node.args) != 1 or not isinstance(node.args[0], Node):
                    raise ValueError("Only a single tensor output is supported")
                lines.append(f"output {node.args[0].name}")
            else:
                raise ValueError(f"Unsupported node kind: {node.op}")
        except (ValueError, RuntimeError, KeyError) as error:
            raise ValueError(f"FX node {node.name}: {error}") from error
    return "\n".join(lines) + "\n"


def export_inputs(graph_text, inputs):
    """Serialize values in the same placeholder order used for export_graph."""
    declarations = [line.split() for line in graph_text.splitlines()
                    if line.startswith("input ")]
    if len(declarations) != len(inputs):
        raise ValueError("Input count does not match exported declarations")
    lines = []
    for declaration, value in zip(declarations, inputs):
        check_tensor(value)
        _, name, rows, columns = declaration
        if list(value.shape) != [int(rows), int(columns)]:
            raise ValueError(f"Wrong shape for input: {name}")
        lines.append(name + " " + " ".join(map(repr, value.detach().flatten().tolist())))
    return "\n".join(lines) + "\n"


class LinearReLU(torch.nn.Module):
    def __init__(self, weights, bias, reverse_add=False):
        super().__init__()
        self.register_buffer("weights", weights)
        self.register_buffer("bias", bias)
        self.reverse_add = reverse_add

    def forward(self, x):
        product = x @ self.weights
        biased = self.bias + product if self.reverse_add else product + self.bias
        return torch.relu(biased)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="File to receive demonstration runtime input values")
    args = parser.parse_args()
    model = LinearReLU(torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([[1., -20.]]))
    inputs = (torch.tensor([[1., 2.]]),)
    text = export_graph(model, inputs)
    data = export_inputs(text, inputs)
    if args.output.resolve() == args.inputs.resolve():
        parser.error("Model and input files must have different paths")
    args.output.write_text(text)
    args.inputs.write_text(data)
    print(f"Exported {args.output} and {args.inputs}")


if __name__ == "__main__":
    main()
