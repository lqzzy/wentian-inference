from pathlib import Path
import os
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_public_layout_has_no_internal_work_documents(self):
        for relative in ("docs", "results", "NOTICE.md"):
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_required_runtime_files_exist(self):
        required = (
            "src/wentian/model/network.py",
            "src/wentian/model/utils.py",
            "src/wentian/runtime/executor.py",
            "src/wentian/runtime/operators.py",
            "src/wentian/runtime/topology.py",
            "src/wentian/data/preprocess.py",
            "src/wentian/data/constants/era5_infer.npz",
            "weights/wentian_beta.pth.gz",
            "run_fp32.sh",
            "run_fp64.sh",
            "scripts/submit_920f.sh",
            "scripts/run_920f.sbatch",
            "scripts/run_920f_job.sh",
            "scripts/ensure_checkpoint.py",
        )
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_cli_is_small_and_precision_aware(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-m", "wentian", "--help"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("{fp32,fp64}", result.stdout)
        self.assertIn("input_root", result.stdout)
        self.assertIn("timestamp", result.stdout)
        for option in ("--checkpoint", "--warmups", "--repeats", "--output-dir", "--threads"):
            self.assertNotIn(option, result.stdout)

    def test_precision_profiles_are_distinct(self):
        environment = os.environ.copy()
        environment.pop("WPH", None)
        environment.pop("WPW", None)
        environment.pop("WNT", None)
        environment.pop("WNT_MAIN", None)
        environment["PYTHONPATH"] = str(ROOT / "src")
        script = (
            "from wentian.runtime.topology import RuntimeTopology; "
            "a=RuntimeTopology.from_environment('fp32', {}); "
            "b=RuntimeTopology.from_environment('fp64', {}); "
            "print(a.height_partitions, a.width_partitions, b.height_partitions, b.width_partitions)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "4 4 3 5")

    def test_submit_scripts_have_no_machine_specific_path(self):
        for relative in ("scripts/submit_920f.sh", "scripts/run_920f.sbatch", "scripts/run_920f_job.sh"):
            content = (ROOT / relative).read_text()
            self.assertNotIn("--nodelist", content)


if __name__ == "__main__":
    unittest.main()
