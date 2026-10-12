"""End-to-end checks for the text graph importer; no third-party packages."""

from pathlib import Path
import subprocess
import sys
import tempfile


runner = str(Path(sys.argv[1]).resolve())
example = Path(__file__).resolve().parents[1] / "example.graph"
result = subprocess.run([runner, str(example)], capture_output=True, text=True, timeout=10)
assert result.returncode == 0, result.stderr
for expected in ("Operations: 3 -> 1", "FusedLinearReLU", "Original result: 8 0",
                 "Optimized result: 8 0", "Outputs match: true"):
    assert expected in result.stdout, result.stdout

invalid = [
    ("tensor x 0 2\n", "Line 1: Expected: tensor"),
    ("tensor x -1 2\n", "Line 1: Expected: tensor"),
    ("tensor x 1 2 1\n", "Line 1: Expected rows * columns"),
    ("tensor x 1 1 nan\n", "Line 1: Expected rows * columns"),
    ("tensor x 1 1 1 extra\n", "Line 1: Unexpected extra argument"),
    ("tensor x 1 1 1\ntensor x 1 1 2\n", "Line 2: Duplicate tensor name"),
    ("\nrelu y missing\n", "Line 2: Unknown tensor: missing"),
    ("tensor x 1 1 1\nadd x x x\n", "Line 2: Duplicate tensor name"),
    ("tensor x 1 1 1\nrelu x x\n", "Line 2: Duplicate tensor name"),
    ("matmul y x\n", "Line 1: Expected: matmul"),
    ("tensor x 1 2 1 2\nmatmul y x x\n", "Line 2:"),
    ("output x extra\n", "Line 1: Unexpected extra argument"),
    ("unknown\n", "Line 1: Unknown opcode"),
    ("tensor x 1 1 1\n", "Missing output instruction"),
    ("tensor x 1 1 1\noutput x\noutput x\n", "Line 3: Only one output"),
]
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "test.graph"
    for program, expected in invalid:
        path.write_text(program)
        result = subprocess.run([runner, str(path)], capture_output=True, text=True, timeout=10)
        assert result.returncode == 1, (program, result)
        assert expected in result.stderr, (program, result.stderr)
        assert not result.stdout, result.stdout
    path.write_text("tensor x 1 2 -3 4\noutput x\n")
    result = subprocess.run([runner, str(path)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "Operations: 0 -> 0" in result.stdout
    assert "Optimized result: -3 4" in result.stdout

print(f"PASS: fused graph, input-only graph, {len(invalid)} invalid graphs")
