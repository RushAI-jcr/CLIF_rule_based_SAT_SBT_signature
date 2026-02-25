import py_compile
import unittest
from pathlib import Path


class NotebookParityTests(unittest.TestCase):
    def test_numbered_notebooks_have_paired_py(self) -> None:
        code_dir = Path(__file__).resolve().parents[1] / "code"
        ipynb_files = sorted(code_dir.glob("0[0-9]_*.ipynb"))
        self.assertGreater(len(ipynb_files), 0)

        missing_pairs = [str(ipynb) for ipynb in ipynb_files if not ipynb.with_suffix(".py").exists()]
        self.assertEqual(missing_pairs, [], msg=f"Missing notebook pairs: {missing_pairs}")

    def test_paired_py_compile(self) -> None:
        code_dir = Path(__file__).resolve().parents[1] / "code"
        py_files = sorted(code_dir.glob("0[0-9]_*.py"))
        self.assertGreater(len(py_files), 0)
        for py_file in py_files:
            py_compile.compile(str(py_file), doraise=True)


if __name__ == "__main__":
    unittest.main()
