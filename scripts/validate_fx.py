"""Validate FX export, C++ import, and fusion against PyTorch."""
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from export_fx import LinearReLU, export_graph


def main():
    runner = str(Path(sys.argv[1]).resolve())
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(42)
    count = 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.graph"
        for m, k, n in [(1, 1, 1), (1, 2, 2), (2, 3, 4), (5, 7, 3)]:
            for reverse in [False, True]:
                x = torch.randn(m, k, generator=generator)
                model = LinearReLU(torch.randn(k, n, generator=generator),
                                   torch.randn(m, n, generator=generator), reverse)
                path.write_text(export_graph(model, (x,)))
                result = subprocess.run([runner, str(path)], capture_output=True,
                                        text=True, timeout=30, check=True)
                assert "Operations: 3 -> 1" in result.stdout, result.stdout
                assert "FusedLinearReLU" in result.stdout, result.stdout
                assert "Outputs match: true" in result.stdout, result.stdout
                expected = model(x)
                for prefix in ["Original result: ", "Optimized result: "]:
                    line = next(line for line in result.stdout.splitlines() if line.startswith(prefix))
                    actual = torch.tensor(list(map(float, line[len(prefix):].split()))).reshape(m, n)
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                count += 1

    class Unsupported(torch.nn.Module):
        def forward(self, x):
            return torch.sigmoid(x)

    class MultipleOutputs(torch.nn.Module):
        def forward(self, x):
            return x, x

    class Method(torch.nn.Module):
        def forward(self, x):
            return x.relu()

    class Scalar(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class Keyword(torch.nn.Module):
        def forward(self, x):
            return torch.add(x, x, alpha=2)

    x = torch.ones(2, 2)
    cases = [(Unsupported(), (x,)), (MultipleOutputs(), (x,)), (Method(), (x,)),
             (Scalar(), (x,)), (Keyword(), (x,)),
             (LinearReLU(x, torch.ones(1, 2)), (x,)),
             (LinearReLU(x, x), (x.double(),)),
             (LinearReLU(x, x), (torch.ones(2),)),
             (LinearReLU(x, x), (torch.full((2, 2), float('nan')),)),
             (LinearReLU(x, x), ())]
    for model, inputs in cases:
        try:
            export_graph(model, inputs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsupported export accepted: {model}")
    print(f"PASS: {count} FX graphs, {2 * count} PyTorch comparisons, {len(cases)} rejected exports")


if __name__ == "__main__":
    main()
