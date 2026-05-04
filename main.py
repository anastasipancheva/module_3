"""
Entry point for Mindly.

  uv run python main.py              — запустить Streamlit UI
  uv run python main.py --eval       — запустить бенчмарк
  uv run python main.py --isolation  — проверить изоляцию тенантов
"""
import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Mindly AI Coach")
    parser.add_argument("--eval", action="store_true", help="Run memory benchmark")
    parser.add_argument("--no-wandb", action="store_true", help="Skip W&B in eval")
    parser.add_argument("--isolation", action="store_true", help="Run tenant isolation test")
    args = parser.parse_args()

    if args.eval:
        cmd = [sys.executable, "scripts/eval_benchmark.py"]
        if args.no_wandb:
            cmd.append("--no-wandb")
        sys.exit(subprocess.call(cmd))

    if args.isolation:
        sys.exit(subprocess.call([sys.executable, "scripts/test_isolation.py"]))

    sys.exit(subprocess.call(["streamlit", "run", "src/app.py"]))


if __name__ == "__main__":
    main()
