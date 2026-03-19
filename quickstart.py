#!/usr/bin/env python3
"""
Guardrail Framework - Quick Start
Run this file to explore all features of the framework.
"""


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          GUARDRAIL FRAMEWORK - QUICK START                  ║
╚══════════════════════════════════════════════════════════════╝
""")

    print("Select mode:")
    print("  1. Interactive CLI")
    print("  2. Run API Server (FastAPI)")
    print("  3. Run Dashboard (Streamlit)")
    print("  4. Run Test Suite")
    print("  5. Demo - Evaluate sample texts")
    print("  6. Demo - Adversarial tests")
    print("  7. Demo - Full integration")
    print("  8. Exit")

    choice = input("\nEnter choice (1-8): ").strip()

    if choice == "1":
        from guardrail_framework import GuardrailCLI
        cli = GuardrailCLI()
        cli.start()

    elif choice == "2":
        from api_server import run_api_server
        print("\nStarting API server at http://localhost:8000")
        print("Docs available at http://localhost:8000/docs")
        run_api_server()

    elif choice == "3":
        import subprocess
        subprocess.run(["streamlit", "run", "dashboard.py"])

    elif choice == "4":
        _run_tests()

    elif choice == "5":
        _demo_evaluate()

    elif choice == "6":
        _demo_adversarial()

    elif choice == "7":
        _demo_integration()

    elif choice == "8":
        print("Goodbye!")
    else:
        print("Invalid choice")


def _run_tests():
    from guardrail_framework import (
        GuardrailEngine, create_default_guardrails,
        create_default_test_cases, ReportGenerator
    )
    engine = GuardrailEngine()
    for rule in create_default_guardrails():
        engine.add_rule(rule)
    for tc in create_default_test_cases():
        engine.add_test_case(tc)

    results = engine.run_tests()
    report = ReportGenerator().generate(results)
    print("\n" + report)

    passed = sum(1 for r in results if r.passed)
    print(f"\n{'PASSED' if passed == len(results) else 'SOME FAILURES'}: {passed}/{len(results)} tests passed")


def _demo_evaluate():
    from guardrail_framework import GuardrailEngine, create_default_guardrails
    engine = GuardrailEngine()
    for rule in create_default_guardrails():
        engine.add_rule(rule)

    test_inputs = [
        "Hello, how are you today?",
        "My SSN is 123-45-6789",
        "Credit card: 4111 1111 1111 1111",
        "I want to end my life",
        "What is the capital of France?",
    ]

    print("\n=== Evaluation Demo ===")
    for text in test_inputs:
        result = engine.evaluate(text)
        icon = "BLOCK" if result.action == "block" else "ALLOW"
        print(f"[{icon}] {text[:50]}..." if len(text) > 50 else f"[{icon}] {text}")
        if result.matched_rules:
            print(f"       Rules: {result.matched_rules}")


def _demo_adversarial():
    from guardrail_framework import GuardrailEngine, create_default_guardrails
    from adversarial_tester import AdversarialTestGenerator

    engine = GuardrailEngine()
    for rule in create_default_guardrails():
        engine.add_rule(rule)

    gen = AdversarialTestGenerator()
    seed = "end my life"
    cases = gen.generate_all(seed, expected_block=True)
    stats = gen.run_against_engine(cases, engine)

    print(f"\n=== Adversarial Test Demo (seed: '{seed}') ===")
    print(f"Total mutations: {stats['total']}")
    print(f"Blocked: {stats['blocked']}")
    print(f"Evaded: {stats['evaded']}")

    if stats["evasions"]:
        print("\nEvasions detected (guardrail gaps to fix):")
        for e in stats["evasions"][:3]:
            print(f"  [{e['mutation_type']}] {e['mutated']}")
    else:
        print("\nNo evasions found - guardrails are robust to these mutations!")


def _demo_integration():
    from guardrail_framework import GuardrailEngine, create_default_guardrails
    from llm_wrapper import GuardedLLM, MockLLMProvider, LLMRequest
    from audit_logger import AuditLogger, create_audit_entry
    from performance_profiler import profiler

    print("\n=== Integration Demo ===")

    # Setup
    engine = GuardrailEngine()
    for rule in create_default_guardrails():
        engine.add_rule(rule)

    provider = MockLLMProvider("This is a helpful response about your question.")
    llm = GuardedLLM(provider, engine)
    audit = AuditLogger(":memory:")

    prompts = [
        "Tell me about Python programming",
        "My SSN is 123-45-6789, help me",
        "What is machine learning?",
    ]

    for prompt in prompts:
        with profiler.time("guarded_llm", "complete"):
            result = llm.complete(LLMRequest(prompt=prompt))

        status = "BLOCKED" if result.blocked else "PASSED"
        print(f"[{status}] {prompt[:50]}")
        if result.blocked:
            print(f"         Reason: {result.block_reason}")
        else:
            print(f"         Response: {result.response.text[:60]}...")

    print("\nLLM Stats:", llm.get_stats())
    print("\nPerformance Stats:", profiler.get_stats("guarded_llm"))


if __name__ == "__main__":
    main()
