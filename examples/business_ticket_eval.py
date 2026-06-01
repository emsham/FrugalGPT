import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from service.modelservice import GenerationParameter, make_model


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
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--spend-cap",
        type=float,
        default=None,
        help="Stop after the cascade reaches this total cost. Useful for live pilots.",
    )
    parser.add_argument(
        "--compare-strong-only",
        action="store_true",
        help="Also call the strong provider for every row to calculate strong-only baseline cost.",
    )
    parser.add_argument("--max-tokens", type=int, default=160)
    return parser.parse_args()


def parse_service_name(service_name):
    if "/" not in service_name:
        raise ValueError(f"Service name must be provider/model, got {service_name!r}")
    return service_name.split("/", 1)


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
):
    genparams = GenerationParameter(max_tokens=max_tokens, temperature=0)
    output_rows = []
    cascade_total = 0.0
    strong_only_total = 0.0

    for row in rows:
        if spend_cap is not None and cascade_total >= spend_cap:
            break

        cheap_payload, cheap_cost, cheap_raw = run_service(
            cheap_service, cheap_service_name, row["text"], genparams
        )
        final_payload = cheap_payload
        raw_completion = cheap_raw
        routed_to = "cheap"
        cascade_cost = cheap_cost

        if should_route_to_strong(cheap_payload, confidence_threshold):
            strong_payload, strong_cost, strong_raw = run_service(
                strong_service, strong_service_name, row["text"], genparams
            )
            final_payload = strong_payload
            raw_completion = strong_raw
            routed_to = "strong"
            cascade_cost += strong_cost

        strong_only_cost = ""
        if compare_strong_only:
            _, baseline_cost, _ = run_service(
                strong_service, strong_service_name, row["text"], genparams
            )
            strong_only_total += baseline_cost
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

        output_rows.append(
            {
                "ticket_id": row.get("ticket_id", ""),
                "text": row["text"],
                "expected_category": row.get("expected_category", ""),
                "expected_urgency": row.get("expected_urgency", ""),
                "expected_escalation": row.get("expected_escalation", ""),
                "predicted_category": final_payload.get("category", ""),
                "predicted_urgency": final_payload.get("urgency", ""),
                "predicted_escalation": str(predicted_escalation).lower(),
                "confidence": final_payload.get("confidence", ""),
                "routed_to": routed_to,
                "cascade_cost": f"{cascade_cost:.8f}",
                "strong_only_cost": strong_only_cost,
                "category_correct": category_correct,
                "urgency_correct": urgency_correct,
                "escalation_correct": escalation_correct,
                "all_correct": all_correct,
                "review_decision": "",
                "reviewer_notes": "",
                "raw_completion": raw_completion,
            }
        )

    return output_rows, cascade_total, strong_only_total


def write_output(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, cascade_total, strong_only_total):
    evaluated = len(rows)
    routed_strong = sum(1 for row in rows if row["routed_to"] == "strong")
    fully_labeled = [row for row in rows if row["all_correct"] != ""]
    all_correct = sum(1 for row in fully_labeled if row["all_correct"] == "true")

    print(f"rows={evaluated}")
    print(f"routed_to_strong={routed_strong}")
    print(f"cascade_cost={cascade_total:.8f}")
    if strong_only_total:
        savings = 1 - (cascade_total / strong_only_total)
        print(f"strong_only_cost={strong_only_total:.8f}")
        print(f"estimated_savings={savings:.1%}")
    if fully_labeled:
        print(f"all_correct_rate={all_correct / len(fully_labeled):.1%}")


def main():
    args = parse_args()
    tickets = read_tickets(args.input, limit=args.limit)
    cheap_service = make_service(args.cheap_provider)
    strong_service = make_service(args.strong_provider)
    rows, cascade_total, strong_only_total = evaluate_rows(
        tickets,
        cheap_service,
        args.cheap_provider,
        strong_service,
        args.strong_provider,
        confidence_threshold=args.confidence_threshold,
        compare_strong_only=args.compare_strong_only,
        spend_cap=args.spend_cap,
        max_tokens=args.max_tokens,
    )
    write_output(args.output, rows)
    summarize(rows, cascade_total, strong_only_total)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
