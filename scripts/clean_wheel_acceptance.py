import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    wheels = sorted(Path("dist").glob("financial_algorithms-0.2.0-*.whl"))
    if len(wheels) != 1:
        raise SystemExit("exactly one 0.2.0 wheel is required")
    with tempfile.TemporaryDirectory() as directory:
        environment = Path(directory) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        python = environment / "bin" / "python"
        subprocess.run([str(python), "-m", "pip", "install", str(wheels[0])], check=True)
        code = """
import json
from financial_algorithms import decide
result=decide({'operation':'market.assess','prices':list(range(100,180))})
print(json.dumps({'engine':result['engine_id'],'version':result['engine_version'],'authorized':result['execution_authorized']}))
"""
        output = subprocess.run(
            [str(python), "-c", code], check=True, text=True, capture_output=True
        )
        parsed = json.loads(output.stdout)
        if parsed != {"engine": "financial-algorithm", "version": "0.2.0", "authorized": False}:
            raise SystemExit("installed-wheel acceptance failed")
    print("clean-wheel acceptance passed")


if __name__ == "__main__":
    main()
