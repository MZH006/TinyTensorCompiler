"""Compile standalone generated programs and compare them against PyTorch."""
import argparse
from pathlib import Path
import subprocess
import tempfile

import torch
from export_fx import LinearReLU, export_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=Path("graph-import"))
    parser.add_argument("--cxx", default="clang++")
    args = parser.parse_args()
    runner = str(args.runner.resolve())
    generator = torch.Generator().manual_seed(17)
    torch.set_num_threads(1)
    comparisons = 0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model_file, source, executable = [root / name for name in ("model.graph", "model.cpp", "model")]
        for m, k, n in [(1, 2, 2), (2, 3, 4), (3, 1, 2)]:
            for reverse in [False, True]:
                model = LinearReLU(torch.randn(k, n, generator=generator),
                                   torch.randn(m, n, generator=generator), reverse)
                x = torch.randn(m, k, generator=generator)
                model_file.write_text(export_graph(model, (x,)))
                subprocess.run([runner, "--emit-cpp", str(model_file), str(source)],
                               capture_output=True, text=True, check=True, timeout=30)
                # Compile only emitted source: no compiler runtime or project headers.
                subprocess.run([args.cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-O3",
                                str(source), "-o", str(executable)],
                               capture_output=True, text=True, check=True, timeout=30)
                for sample in [x, torch.randn(m, k, generator=generator)]:
                    payload = " ".join(map(repr, sample.flatten().tolist())) + "\n"
                    result = subprocess.run([str(executable)], input=payload,
                                            capture_output=True, text=True, check=True, timeout=10)
                    actual = torch.tensor(list(map(float, result.stdout.split()))).reshape(m, n)
                    torch.testing.assert_close(actual, model(sample), rtol=1e-4, atol=1e-5)
                    comparisons += 1
                for payload in ["", "nan\n", " ".join(["0"] * (m * k + 1))]:
                    result = subprocess.run([str(executable)], input=payload,
                                            capture_output=True, text=True, timeout=10)
                    assert result.returncode == 1 and result.stderr and not result.stdout, result
        # Unsupported structures must fail without creating source.
        source.unlink()
        model_file.write_text("input x 1 2\nrelu y x\noutput y\n")
        result = subprocess.run([runner, "--emit-cpp", str(model_file), str(source)],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 1 and not source.exists(), result
    print(f"PASS: 6 standalone programs, {comparisons} PyTorch comparisons, "
          "18 invalid runtime inputs and unsupported graph rejection")


if __name__ == "__main__":
    main()
