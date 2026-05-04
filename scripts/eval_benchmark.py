"""
Mini LongMemEval-style benchmark for MindlyAgent.

Methodology:
  - Mimics LongMemEval (Wu et al., 2024): inject facts into the agent's memory
    via simulated conversations, then ask recall questions and grade answers.
  - Metric: soft Exact-Match (EM) — answer considered correct if the expected
    key phrase appears anywhere in the model's reply (case-insensitive).
  - Dataset: 25 handcrafted QA pairs in the spirit of LongMemEval's categories:
    single-hop fact recall, temporal reasoning, and preference recall.

Usage:
    cd <project-root>
    python scripts/eval_benchmark.py [--no-wandb]

Results are logged to:
  · W&B run (project mindly-eval)
  · data/log_file.log
  · stdout
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import MindlyAgent  # noqa: E402
from loguru import logger

EVAL_CASES: list[tuple[str, list[str], str, str]] = [
    (
        "eval_u1",
        ["My name is Dmitry and I'm 28 years old."],
        "How old am I?",
        "28",
    ),
    (
        "eval_u1",
        ["I work as a backend engineer at a fintech startup."],
        "What's my profession?",
        "backend",
    ),
    (
        "eval_u1",
        ["My current weight is 94 kg."],
        "What is my weight?",
        "94",
    ),
    (
        "eval_u1",
        ["My goal is to lose 12 kg before December."],
        "What is my fitness goal?",
        "12",
    ),
    (
        "eval_u2",
        ["I have a dog named Bruno."],
        "What is my dog's name?",
        "Bruno",
    ),
    (
        "eval_u2",
        ["I'm vegetarian and I don't eat meat."],
        "Am I vegetarian?",
        "vegetarian",
    ),
    (
        "eval_u2",
        ["I wake up at 6:30 every morning to exercise."],
        "What time do I wake up?",
        "6:30",
    ),
    (
        "eval_u2",
        ["I'm studying Spanish and want to reach B2 level."],
        "What language am I learning?",
        "Spanish",
    ),
    (
        "eval_u3",
        ["I prefer morning workouts, I hate going to the gym in the evening."],
        "When do I prefer to work out?",
        "morning",
    ),
    (
        "eval_u3",
        ["I love hiking and try to go at least once a month."],
        "What outdoor activity do I enjoy?",
        "hiking",
    ),
    (
        "eval_u3",
        ["My biggest motivation is my daughter — I want to be healthy for her."],
        "What motivates me the most?",
        "daughter",
    ),
    (
        "eval_u3",
        ["I struggle with stress eating, especially at night."],
        "What eating challenge do I have?",
        "stress",
    ),
    (
        "eval_u4",
        ["I drink about 1 litre of water a day, which I know is too little."],
        "How much water do I drink daily?",
        "1",
    ),
    (
        "eval_u4",
        ["I sleep only 5–6 hours a night because of my baby."],
        "How many hours do I sleep?",
        "5",
    ),
    (
        "eval_u4",
        ["I've been doing intermittent fasting 16:8 for two months."],
        "What diet pattern do I follow?",
        "intermittent",
    ),
    (
        "eval_u5",
        [
            "I started going to the gym three weeks ago.",
            "I've been going 3 times a week since I started.",
        ],
        "How often do I go to the gym?",
        "3",
    ),
    (
        "eval_u5",
        [
            "Last week I ran 5 km without stopping for the first time.",
            "My previous personal best was 3 km.",
        ],
        "What is my latest running achievement?",
        "5",
    ),
    (
        "eval_u5",
        [
            "I had a bad week — I skipped all my workouts.",
            "The reason was a business trip to Moscow.",
        ],
        "Why did I skip my workouts?",
        "trip",
    ),
    (
        "eval_u6",
        ["I have a knee injury that makes running painful."],
        "What injury do I have?",
        "knee",
    ),
    (
        "eval_u6",
        ["My doctor told me to avoid high-impact exercises for 6 weeks."],
        "What did my doctor advise?",
        "6",
    ),
    (
        "eval_u6",
        ["I'd like to try swimming as a low-impact alternative."],
        "What exercise am I considering?",
        "swimming",
    ),
    (
        "eval_u7",
        ["I want to do 50 push-ups in a row by the end of the month."],
        "What's my push-up goal?",
        "50",
    ),
    (
        "eval_u7",
        ["Currently I can do 20 push-ups."],
        "How many push-ups can I do now?",
        "20",
    ),
    (
        "eval_u7",
        ["I track my calories with an app and aim for 1800 kcal per day."],
        "What is my daily calorie target?",
        "1800",
    ),
    (
        "eval_u8",
        ["My blood pressure has been high — 150/95 last check."],
        "What was my last blood pressure reading?",
        "150",
    ),
]

EVAL_PERSONA = "wellness-friend"


def soft_em(answer: str, expected: str) -> bool:
    return expected.lower() in answer.lower()


def run_eval(agent: MindlyAgent) -> dict:
    results = []

    all_user_ids = {case[0] for case in EVAL_CASES}
    logger.info("eval_start | users={} cases={}", len(all_user_ids), len(EVAL_CASES))

    for uid in all_user_ids:
        agent.forget(uid, "all")

    for i, (user_id, setups, question, expected) in enumerate(EVAL_CASES):
        for setup_msg in setups:
            agent.chat(user_id, EVAL_PERSONA, setup_msg, stream=False)
            time.sleep(0.3) 

        t0 = time.perf_counter()
        answer = agent.chat(user_id, EVAL_PERSONA, question, stream=False)
        latency_ms = (time.perf_counter() - t0) * 1000

        correct = soft_em(answer, expected)
        results.append(
            {
                "idx": i,
                "user_id": user_id,
                "question": question,
                "expected": expected,
                "answer": answer[:200],
                "correct": correct,
                "latency_ms": latency_ms,
            }
        )

        status = "✓" if correct else "✗"
        print(
            f"[{i+1:02d}/{len(EVAL_CASES)}] {status} "
            f"Q: {question[:50]:50s} | expected='{expected}' | "
            f"ans='{answer[:40]}' | {latency_ms:.0f}ms"
        )
        logger.info(
            "eval_case | idx={} user={} correct={} latency_ms={:.0f} q='{}' expected='{}' ans='{}'",
            i, user_id, correct, latency_ms, question[:60], expected, answer[:60],
        )

        time.sleep(0.5)

    n_correct = sum(r["correct"] for r in results)
    em_score = n_correct / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    summary = {
        "em_score": em_score,
        "n_correct": n_correct,
        "n_total": len(results),
        "avg_latency_ms": avg_latency,
        "results": results,
    }
    logger.info(
        "eval_done | em={:.3f} ({}/{}) avg_latency_ms={:.0f}",
        em_score, n_correct, len(results), avg_latency,
    )
    return summary


def log_to_wandb(summary: dict) -> None:
    try:
        import wandb

        run = wandb.init(
            project="mindly-eval",
            name=f"eval-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "benchmark": "MiniLongMemEval-v1",
                "n_cases": summary["n_total"],
                "persona": EVAL_PERSONA,
                "model": os.getenv("CHAT_MODEL", "openai/gpt-4o-mini"),
                "mem0_llm": os.getenv("MEM0_LLM_MODEL", "openai/gpt-4o-mini"),
                "embedder": "sentence-transformers/all-MiniLM-L6-v2",
                "vector_store": "chromadb",
                "memory_framework": "mem0",
            },
        )

        wandb.log(
            {
                "em_score": summary["em_score"],
                "n_correct": summary["n_correct"],
                "n_total": summary["n_total"],
                "avg_latency_ms": summary["avg_latency_ms"],
            }
        )

        table = wandb.Table(
            columns=["idx", "user_id", "question", "expected", "answer", "correct", "latency_ms"]
        )
        for r in summary["results"]:
            table.add_data(
                r["idx"], r["user_id"], r["question"],
                r["expected"], r["answer"], r["correct"], r["latency_ms"],
            )
        wandb.log({"eval_table": table})
        run.finish()
        print(f"\nW&B run: {run.url}")
    except ImportError:
        print("wandb not installed — skipping W&B logging. Run: uv add wandb")
    except Exception as exc:
        print(f"W&B logging failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mindly memory benchmark")
    parser.add_argument("--no-wandb", action="store_true", help="Skip W&B logging")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  Mindly Mini-LongMemEval Benchmark")
    print("=" * 70)
    print(f"  Cases   : {len(EVAL_CASES)}")
    print(f"  Persona : {EVAL_PERSONA}")
    print(f"  Model   : {os.getenv('CHAT_MODEL', 'openai/gpt-4o-mini')}")
    print("=" * 70 + "\n")

    agent = MindlyAgent()
    summary = run_eval(agent)

    print("\n" + "=" * 70)
    print(f"  EM Score  : {summary['em_score']:.1%}  ({summary['n_correct']}/{summary['n_total']})")
    print(f"  Avg latency: {summary['avg_latency_ms']:.0f} ms")
    print("=" * 70)

    if not args.no_wandb:
        log_to_wandb(summary)


if __name__ == "__main__":
    main()
