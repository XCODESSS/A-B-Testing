import os
import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_all_four_pages_load_without_runtime_errors():
    python_executable = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    assert python_executable.exists(), "Expected project virtualenv python executable."

    temp_root = PROJECT_ROOT / ".tmp_runtime"
    temp_root.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["TEMP"] = str(temp_root)
    env["TMP"] = str(temp_root)

    script = """
from pathlib import Path
from streamlit.testing.v1 import AppTest

app_path = Path('app.py').resolve()
at = AppTest.from_file(str(app_path))
at.run(timeout=10)
assert not at.exception, at.exception
radio = at.sidebar.radio[0]
expected = ['Experiment Planning', 'Statistical Analysis', 'Pitfall Detection', 'Business Impact']
assert list(radio.options) == expected, radio.options
for page in expected:
    radio.set_value(page)
    at.run(timeout=10)
    assert not at.exception, f'{page}: {at.exception}'
"""

    result = subprocess.run(
        [str(python_executable), "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
