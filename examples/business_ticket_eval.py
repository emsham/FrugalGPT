import argparse
import csv
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from service.modelservice import GenerationParameter, make_model


PROVIDER_PRESETS = {
    "fake": ("fake/support-cheap", "fake/support-strong"),
    "openai": ("openaichat/gpt-4o-mini", "openaichat/gpt-4o"),
    "anthropic": (
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3-5-sonnet-20240620",
    ),
    "mixed-openai-anthropic": (
        "openaichat/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet-20240620",
    ),
}

REQUIRED_ENV = {
    "ai21": "AI21_STUDIO_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "COHERE_STUDIO_API_KEY",
    "forefrontai": "FOREFRONT_API_KEY",
    "google": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openaichat": "OPENAI_API_KEY",
    "textsynth": "TEXTSYNTH_API_SECRET_KEY",
    "togetherai": "TOGETHER_API_KEY",
}

OUTPUT_FIELDS = [
    "ticket_id",
    "text",
    "expected_category",
    "expected_urgency",
    "expected_escalation",
    "predicted_category",
    "predicted_urgency",
    "predicted_escalation",
    "confidence",
    "routed_to",
    "status",
    "error_message",
    "cascade_cost",
    "strong_only_cost",
    "category_correct",
    "urgency_correct",
    "escalation_correct",
    "all_correct",
    "review_decision",
    "reviewer_notes",
    "raw_completion",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a cheap-first FrugalGPT-style cascade on business tickets."
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "examples" / "business_tickets_sample.csv"),
        help="Input CSV with ticket_id,text and optional expected_* columns.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "examples" / "business_ticket_eval_output.csv"),
        help="Output CSV path for model decisions and human review columns.",
    )
    parser.add_argument("--cheap-provider", default="fake/support-cheap")
    parser.add_argument("--strong-provider", default="fake/support-strong")
    parser.add_argument(
        "--provider-preset",
        choices=sorted(PROVIDER_PRESETS),
        default=None,
        help="Convenience preset for cheap/strong providers. Explicit provider args override it.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--spend-cap",
        type=float,
        default=None,
        help=(
            "Stop before starting a new ticket or extra strong call once observed "
            "cascade cost reaches this amount."
        ),
    )
    parser.add_argument(
        "--compare-strong-only",
        action="store_true",
        help="Also call the strong provider for every row to calculate strong-only baseline cost.",
    )
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort on the first row/provider error instead of writing an error row.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned run without making provider calls.",
    )
    return parser.parse_args()


def parse_service_name(service_name):
    if "/" not in service_name:
        raise ValueError(f"Service name must be provider/model, got {service_name!r}")
    return service_name.split("/", 1)


def resolve_provider_names(args):
    if args.provider_preset:
        preset_cheap, preset_strong = PROVIDER_PRESETS[args.provider_preset]
        cheap_provider = (
            args.cheap_provider
            if args.cheap_provider != "fake/support-cheap"
            else preset_cheap
        )
        strong_provider = (
            args.strong_provider
            if args.strong_provider != "fake/support-strong"
            else preset_strong
        )
        return cheap_provider, strong_provider
    return args.cheap_provider, args.strong_provider


def required_env_vars(*service_names):
    env_vars = []
    for service_name in service_names:
        provider, _ = parse_service_name(service_name)
        env_var = REQUIRED_ENV.get(provider)
        if env_var and env_var not in env_vars:
            env_vars.append(env_var)
    return env_vars


def make_service(service_name):
    provider, model = parse_service_name(service_name)
    return make_model(provider, model)


def parse_bool(value):
    if value is None or value == "":
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def norm_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def maybe_correct(expected, predicted):
    if expected in (None, ""):
        return ""
    return str(norm_text(expected) == norm_text(predicted)).lower()


def maybe_bool_correct(expected, predicted):
    if expected is None:
        return ""
    return str(bool(expected) == bool(predicted)).lower()


def build_prompt(ticket_text, service_name):
    if service_name.startswith("fake/"):
        return ticket_text
    return (
        "Classify the following business support ticket. Return only JSON with keys "
        "category, urgency, escalation, confidence, summary, and next_action. "
        "Use category values billing, sales, technical, or general. "
        "Use urgency values high or normal.\n\n"
        f"Ticket: {ticket_text}"
    )


def run_service(service, service_name, ticket_text, genparams):
    result = service.getcompletion(build_prompt(ticket_text, service_name), genparams=genparams)
    try:
        payload = json.loads(result["completion"])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{service_name} did not return JSON for ticket text {ticket_text!r}: "
            f"{result['completion']!r}"
        ) from exc
    return payload, result["cost"], result["completion"]


def should_route_to_strong(payload, confidence_threshold):
    confidence = float(payload.get("confidence", 0))
    return confidence < confidence_threshold or bool(payload.get("escalation", False))


def base_output_row(row):
    return {
        "ticket_id": row.get("ticket_id", ""),
        "text": row["text"],
        "expected_category": row.get("expected_category", ""),
        "expected_urgency": row.get("expected_urgency", ""),
        "expected_escalation": row.get("expected_escalation", ""),
        "predicted_category": "",
        "predicted_urgency": "",
        "predicted_escalation": "",
        "confidence": "",
        "routed_to": "",
        "status": "",
        "error_message": "",
        "cascade_cost": "",
        "strong_only_cost": "",
        "category_correct": "",
        "urgency_correct": "",
        "escalation_correct": "",
        "all_correct": "",
        "review_decision": "",
        "reviewer_notes": "",
        "raw_completion": "",
    }


def error_output_row(row, error):
    output = base_output_row(row)
    output["routed_to"] = "error"
    output["status"] = "error"
    output["error_message"] = str(error)
    return output


def read_tickets(path, limit=None):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    if limit is not None:
        rows = rows[:limit]
    for idx, row in enumerate(rows, start=1):
        if not row.get("text"):
            raise ValueError(f"Row {idx} is missing required text column")
        row.setdefault("ticket_id", str(idx))
    return rows


def evaluate_rows(
    rows,
    cheap_service,
    cheap_service_name,
    strong_service,
    strong_service_name,
    confidence_threshold,
    compare_strong_only=False,
    spend_cap=None,
    max_tokens=160,
    stop_on_error=False,
):
    genparams = GenerationParameter(max_tokens=max_tokens, temperature=0)
    output_rows = []
    cascade_total = 0.0
    strong_only_total = 0.0
    baseline_cascade_total = 0.0

    for row in rows:
        if spend_cap is not None and cascade_total >= spend_cap:
            break

        try:
            cheap_payload, cheap_cost, cheap_raw = run_service(
                cheap_service, cheap_service_name, row["text"], genparams
            )
            final_payload = cheap_payload
            raw_completion = cheap_raw
            routed_to = "cheap"
            status = "ok"
            error_message = ""
            cascade_cost = cheap_cost
            strong_cost_for_row = None

            route_to_strong = should_route_to_strong(cheap_payload, confidence_threshold)
            cap_blocks_extra_call = (
                spend_cap is not None and cascade_total + cascade_cost >= spend_cap
            )

            if route_to_strong and not cap_blocks_extra_call:
                strong_payload, strong_cost, strong_raw = run_service(
                    strong_service, strong_service_name, row["text"], genparams
                )
                strong_cost_for_row = strong_cost
                final_payload = strong_payload
                raw_completion = strong_raw
                routed_to = "strong"
                cascade_cost += strong_cost
            elif route_to_strong and cap_blocks_extra_call:
                routed_to = "cheap_spend_cap"
                status = "spend_cap_hold"
                error_message = (
                    "Skipped strong-provider escalation because the spend cap was "
                    "reached after the cheap-provider call."
                )

            strong_only_cost = ""
            if compare_strong_only:
                if strong_cost_for_row is not None:
                    baseline_cost = strong_cost_for_row
                elif spend_cap is not None and cascade_total + cascade_cost >= spend_cap:
                    baseline_cost = None
                    if not error_message:
                        error_message = "Skipped strong-only baseline because spend cap was reached."
                else:
                    _, baseline_cost, _ = run_service(
                        strong_service, strong_service_name, row["text"], genparams
                    )
                if baseline_cost is not None:
                    strong_only_total += baseline_cost
                    baseline_cascade_total += cascade_cost
                    strong_only_cost = f"{baseline_cost:.8f}"

            cascade_total += cascade_cost
            expected_escalation = parse_bool(
                row.get("expected_escalation", row.get("expected_escalate", ""))
            )
            predicted_escalation = bool(final_payload.get("escalation", False))
            category_correct = maybe_correct(
                row.get("expected_category"), final_payload.get("category")
            )
            urgency_correct = maybe_correct(row.get("expected_urgency"), final_payload.get("urgency"))
            escalation_correct = maybe_bool_correct(expected_escalation, predicted_escalation)
            correctness = [category_correct, urgency_correct, escalation_correct]
            all_correct = (
                str(all(value == "true" for value in correctness)).lower()
                if all(value != "" for value in correctness)
                else ""
            )

            output = base_output_row(row)
            output.update(
                {
                    "predicted_category": final_payload.get("category", ""),
                    "predicted_urgency": final_payload.get("urgency", ""),
                    "predicted_escalation": str(predicted_escalation).lower(),
                    "confidence": final_payload.get("confidence", ""),
                    "routed_to": routed_to,
                    "status": status,
                    "error_message": error_message,
                    "cascade_cost": f"{cascade_cost:.8f}",
                    "strong_only_cost": strong_only_cost,
                    "category_correct": category_correct,
                    "urgency_correct": urgency_correct,
                    "escalation_correct": escalation_correct,
                    "all_correct": all_correct,
                    "raw_completion": raw_completion,
                }
            )
            output_rows.append(output)
        except Exception as exc:
            if stop_on_error:
                raise
            output_rows.append(error_output_row(row, exc))

    return output_rows, cascade_total, strong_only_total, baseline_cascade_total


def write_output(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, cascade_total, strong_only_total, baseline_cascade_total=0.0):
    evaluated = len(rows)
    routed_strong = sum(1 for row in rows if row["routed_to"] == "strong")
    failures = sum(1 for row in rows if row["status"] == "error")
    spend_cap_holds = sum(1 for row in rows if row["status"] == "spend_cap_hold")
    fully_labeled = [row for row in rows if row["all_correct"] != ""]
    all_correct = sum(1 for row in fully_labeled if row["all_correct"] == "true")

    print(f"rows={evaluated}")
    print(f"routed_to_strong={routed_strong}")
    print(f"failures={failures}")
    print(f"spend_cap_holds={spend_cap_holds}")
    print(f"cascade_cost={cascade_total:.8f}")
    if strong_only_total:
        savings = 1 - (baseline_cascade_total / strong_only_total)
        print(f"strong_only_cost={strong_only_total:.8f}")
        if baseline_cascade_total != cascade_total:
            print(f"baseline_matched_cascade_cost={baseline_cascade_total:.8f}")
        print(f"estimated_savings={savings:.1%}")
    if fully_labeled:
        print(f"all_correct_rate={all_correct / len(fully_labeled):.1%}")


def print_dry_run(args, tickets, cheap_provider, strong_provider):
    env_vars = required_env_vars(cheap_provider, strong_provider)
    missing_env = [name for name in env_vars if not os.environ.get(name)]

    print("dry_run=true")
    print(f"rows={len(tickets)}")
    print(f"cheap_provider={cheap_provider}")
    print(f"strong_provider={strong_provider}")
    print(f"confidence_threshold={args.confidence_threshold}")
    print(f"spend_cap={args.spend_cap}")
    print(f"compare_strong_only={str(args.compare_strong_only).lower()}")
    if env_vars:
        print(f"required_env={','.join(env_vars)}")
    if missing_env:
        print(f"missing_env={','.join(missing_env)}")
    print(f"output={args.output}")


def main():
    args = parse_args()
    tickets = read_tickets(args.input, limit=args.limit)
    cheap_provider, strong_provider = resolve_provider_names(args)
    if args.dry_run:
        print_dry_run(args, tickets, cheap_provider, strong_provider)
        return

    cheap_service = make_service(cheap_provider)
    strong_service = make_service(strong_provider)
    rows, cascade_total, strong_only_total, baseline_cascade_total = evaluate_rows(
        tickets,
        cheap_service,
        cheap_provider,
        strong_service,
        strong_provider,
        confidence_threshold=args.confidence_threshold,
        compare_strong_only=args.compare_strong_only,
        spend_cap=args.spend_cap,
        max_tokens=args.max_tokens,
        stop_on_error=args.stop_on_error,
    )
    write_output(args.output, rows)
    summarize(rows, cascade_total, strong_only_total, baseline_cascade_total)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
