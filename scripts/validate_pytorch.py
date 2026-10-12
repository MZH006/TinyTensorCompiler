#!/usr/bin/env python3
"""Validate the fixed C++ linear/ReLU runner against CPU float32 PyTorch."""
import argparse
import json
from pathlib import Path
import subprocess

import torch


def run_cpp(runner, mode, payload, shape, reverse_add):
    command = [str(runner), mode]
    if reverse_add:
        command.append("--reverse-add")
    result = subprocess.run(command, input=payload, text=True, capture_output=True,
                            check=True, timeout=30)
    output = json.loads(result.stdout)
    if output["shape"] != list(shape):
        raise AssertionError(f"Wrong output shape: {output['shape']}")
    expected_count = 1 if mode == "--fused" else 3
    if output["operation_count"] != expected_count:
        raise AssertionError(f"Wrong operation count in {mode}: {output}")
    if len(output["data"]) != shape[0] * shape[1]:
        raise AssertionError("Wrong number of output elements")
    return torch.tensor(output["data"], dtype=torch.float32).reshape(shape)


def validate_case(runner, x, weights, bias, reverse_add=False):
    m, k = x.shape
    n = weights.shape[1]
    # Python repr round-trips the underlying float32 values without truncation.
    values = x.flatten().tolist() + weights.flatten().tolist() + bias.flatten().tolist()
    payload = f"{m} {k} {n}\n" + " ".join(map(repr, values)) + "\n"
    expected = torch.relu(x @ weights + bias)
    for mode in ("--unfused", "--fused"):
        actual = run_cpp(runner, mode, payload, (m, n), reverse_add)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def validate_bad_inputs(runner):
    cases = [([], ""), (["--unknown"], ""), (["--fused"], "-1 2 2"),
             (["--fused"], "257 1 1"), (["--fused"], "1 1 1 2"),
             (["--fused"], "1 1 1 2 3 4 extra"),
             (["--fused"], "1 1 1 nan 3 4")]
    for arguments, payload in cases:
        result = subprocess.run([str(runner), *arguments], input=payload,
                                text=True, capture_output=True, timeout=30)
        if result.returncode != 1 or not result.stderr or result.stdout:
            raise AssertionError(f"Invalid input not cleanly rejected: {arguments}, {payload}")
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path,
                        default=Path(__file__).resolve().parent.parent / "linear-runner")
    args = parser.parse_args()
    runner = args.runner.resolve()
    if not runner.is_file():
        parser.error(f"Build the runner first: make linear-runner ({runner})")
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(42)
    validate_case(runner, torch.tensor([[1., 2.]]),
                  torch.tensor([[3., 4.], [5., 6.]]), torch.tensor([[1., -20.]]))
    validate_case(runner, torch.zeros(2, 3), torch.zeros(3, 2),
                  torch.tensor([[-1., 0.], [1., -2.]]), reverse_add=True)
    shapes = [(1, 1, 1), (1, 7, 3), (3, 1, 5), (2, 3, 4),
              (8, 16, 8), (17, 9, 11), (32, 64, 32), (64, 128, 64)]
    count = 2
    for shape in shapes:
        m, k, n = shape
        for repetition in range(4):
            x = torch.randn(m, k, generator=generator)
            weights = torch.randn(k, n, generator=generator)
            bias = torch.randn(m, n, generator=generator)
            validate_case(runner, x, weights, bias, reverse_add=repetition % 2 == 1)
            count += 1
    rejected = validate_bad_inputs(runner)
    print(f"PASS {count} cases in both modes ({2 * count} comparisons), "
          f"{rejected} invalid-input checks; PyTorch {torch.__version__} CPU float32")


if __name__ == "__main__":
    main()
