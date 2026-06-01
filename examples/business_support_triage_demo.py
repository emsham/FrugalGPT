import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from service.modelservice import GenerationParameter, make_model


@dataclass
class Ticket:
    ticket_id: str
    text: str


TICKETS = [
    Ticket("T-1001", "Customer reports a duplicate charge and asks for a refund."),
    Ticket("T-1002", "Prospect wants pricing for 80 seats and a security review."),
    Ticket("T-1003", "The API integration is returning 500 errors after deploy."),
    Ticket("T-1004", "Customer asks how to change their notification email."),
    Ticket("T-1005", "Enterprise customer says login is down and they may cancel."),
]


def run_model(model, ticket):
    result = model.getcompletion(
        ticket.text,
        genparams=GenerationParameter(max_tokens=120, temperature=0),
    )
    return json.loads(result["completion"]), result["cost"]


def main():
    cheap = make_model("fake", "support-cheap")
    strong = make_model("fake", "support-strong")
    confidence_threshold = 0.7

    cascade_cost = 0
    strong_only_cost = 0
    rows = []

    for ticket in TICKETS:
        cheap_payload, cheap_cost = run_model(cheap, ticket)
        strong_payload, strong_cost = run_model(strong, ticket)
        strong_only_cost += strong_cost

        use_strong = cheap_payload["confidence"] < confidence_threshold or cheap_payload["escalation"]
        final_payload = strong_payload if use_strong else cheap_payload
        final_cost = cheap_cost + strong_cost if use_strong else cheap_cost
        cascade_cost += final_cost

        rows.append(
            {
                "ticket": ticket.ticket_id,
                "category": final_payload["category"],
                "urgency": final_payload["urgency"],
                "routed_to": "strong" if use_strong else "cheap",
                "confidence": final_payload["confidence"],
                "cost": final_cost,
            }
        )

    print("ticket,category,urgency,routed_to,confidence,cost")
    for row in rows:
        print(
            f"{row['ticket']},{row['category']},{row['urgency']},"
            f"{row['routed_to']},{row['confidence']:.2f},{row['cost']:.8f}"
        )

    savings = 1 - (cascade_cost / strong_only_cost)
    print()
    print(f"cascade_cost={cascade_cost:.8f}")
    print(f"strong_only_cost={strong_only_cost:.8f}")
    print(f"estimated_savings={savings:.1%}")


if __name__ == "__main__":
    main()
