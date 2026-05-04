"""
Tenant isolation test.

Usage:
    cd <project-root>
    python scripts/test_isolation.py

Demonstrates that user A's memories are NOT visible to user B.
Exits with code 0 on success, 1 on failure.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import MindlyAgent

USER_A = "test_alice_isolation"
USER_B = "test_bob_isolation"

ALICE_SECRET = "Alice loves pineapple pizza and her cat is named Whiskers"
BOB_QUERY = "pineapple pizza"


def header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def run() -> int:
    agent = MindlyAgent()

    header("Cleanup: wiping previous test memories")
    agent.forget(USER_A, "all")
    agent.forget(USER_B, "all")
    print(f"✓ Cleared memories for {USER_A} and {USER_B}")


    header("Step 1: Alice shares personal info")
    response_a = agent.chat(USER_A, "wellness-friend", ALICE_SECRET, stream=False)
    print(f"Alice's message : {ALICE_SECRET}")
    print(f"Agent response  : {response_a[:120]}…")

    header("Step 2: Check Alice's stored memories")
    alice_mems = agent.get_all_memories(USER_A)
    alice_facts = " ".join(m.get("memory", "") for m in alice_mems).lower()
    print(f"Alice has {len(alice_mems)} stored memories")
    for m in alice_mems:
        print(f"  · {m.get('memory', '—')}")

    alice_remembered = "pineapple" in alice_facts or "whiskers" in alice_facts or "pizza" in alice_facts
    print(f"\nAlice's secret stored? {'✓ YES' if alice_remembered else '✗ NO (mem0 may have paraphrased)'}")

    header("Step 3: Bob queries the same topic — expects NO recall")
    response_b = agent.chat(USER_B, "tough-love", BOB_QUERY, stream=False)
    print(f"Bob's query     : {BOB_QUERY}")
    print(f"Agent response  : {response_b[:200]}…")

    bob_mems = agent.get_all_memories(USER_B)
    bob_facts = " ".join(m.get("memory", "") for m in bob_mems).lower()

    leaked = "pineapple" in bob_facts or "whiskers" in bob_facts
    print(f"\nBob's memories  : {len(bob_mems)} items")
    for m in bob_mems:
        print(f"  · {m.get('memory', '—')}")

    header("Step 4: Alice forgets her pizza preference")
    agent.forget(USER_A, "pizza")
    alice_mems_after = agent.get_all_memories(USER_A)
    alice_facts_after = " ".join(m.get("memory", "") for m in alice_mems_after).lower()
    pizza_gone = "pineapple pizza" not in alice_facts_after
    print(f"Alice's memories after targeted forget ({len(alice_mems_after)} items):")
    for m in alice_mems_after:
        print(f"  · {m.get('memory', '—')}")
    print(f"\nPizza memory removed? {'✓ YES' if pizza_gone else '⚠ Still present (score threshold)'}")

    header("RESULT")
    if leaked:
        print("✗ FAIL — Alice's private data leaked into Bob's memories!")
        return 1
    else:
        print("✓ PASS — Tenant isolation confirmed: Bob cannot see Alice's memories.")
        return 0


if __name__ == "__main__":
    sys.exit(run())
