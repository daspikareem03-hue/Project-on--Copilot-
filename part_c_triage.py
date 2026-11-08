
from pathlib import Path
import csv
import json
import re
import random
import time
from collections import Counter

from mock_llm import chat, reset, TransientLLMError, est_tokens


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"

OUTPUTS.mkdir(parents=True, exist_ok=True)

TICKETS_PATH = RAW / "tickets.csv"
SEGMENTS_PATH = OUTPUTS / "customer_segments.csv"

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3

TOKEN_BUDGET_DEMO = 6000
TOKEN_BUDGET_FINAL = 100000

CATEGORIES = ["billing", "delivery", "technical", "fraud"]


# ============================================================
# PRIVACY REDACTION
# Apply patterns in the required order
# ============================================================

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")

redaction_counts = Counter()


def redact_text(text):
    text, emails = EMAIL_PATTERN.subn("[EMAIL]", text)
    text, phones = PHONE_PATTERN.subn("[PHONE]", text)
    text, orders = ORDER_PATTERN.subn("[ORDER_ID]", text)

    redaction_counts["email"] += emails
    redaction_counts["phone"] += phones
    redaction_counts["order_id"] += orders

    # Assertion: no private identifier remains
    assert not EMAIL_PATTERN.search(text)
    assert not PHONE_PATTERN.search(text)
    assert not ORDER_PATTERN.search(text)

    return text


# ============================================================
# PROMPTS
# ============================================================

ZERO_SHOT_SYSTEM = """
CLASSIFY customer support tickets.

Allowed categories: billing, delivery, technical, fraud.

Return JSON only, exactly in this form:
{"category": "...", "reason": "..."}
"""

FEW_SHOT_SYSTEM = """
CLASSIFY customer support tickets.

Allowed categories: billing, delivery, technical, fraud.

Return JSON only, exactly in this form:
{"category": "...", "reason": "..."}

EXAMPLES:
Example 1:
Ticket: My account was hacked.
Answer: {"category": "fraud", "reason": "account compromise"}

Example 2:
Ticket: My payment was charged twice.
Answer: {"category": "billing", "reason": "duplicate payment"}

Example 3:
Ticket: The app keeps crashing.
Answer: {"category": "technical", "reason": "app crash"}

Example 4:
Ticket: My delivery is late.
Answer: {"category": "delivery", "reason": "late delivery"}
"""


# ============================================================
# LOAD INPUT DATA
# ============================================================

with open(TICKETS_PATH, "r", encoding="utf-8-sig", newline="") as f:
    tickets = list(csv.DictReader(f))

print("Tickets loaded:", len(tickets))


# Redact all tickets once and retain the original text for guardrails.
prepared_tickets = []

for ticket in tickets:
    original_text = ticket["text"] or ""
    safe_text = redact_text(original_text)

    prepared_tickets.append({
        **ticket,
        "original_text": original_text,
        "safe_text": safe_text,
    })


print("\nRedaction totals:")
print("Emails:", redaction_counts["email"])
print("Phones:", redaction_counts["phone"])
print("Order IDs:", redaction_counts["order_id"])
print("Total:", sum(redaction_counts.values()))

assert redaction_counts["email"] == 55
assert redaction_counts["phone"] == 91
assert redaction_counts["order_id"] == 132
assert sum(redaction_counts.values()) == 278


# ============================================================
# LOAD CUSTOMER SEGMENTS FROM PART B
# ============================================================

segments = {}

with open(SEGMENTS_PATH, "r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        segments[row["customer_id"]] = row


def is_premium(customer_id):
    row = segments.get(str(customer_id))

    if not row:
        return False

    value = str(row.get("is_premium", "")).strip().lower()
    return value in {"true", "1", "yes"}


def segment_name(customer_id):
    row = segments.get(str(customer_id))

    if not row:
        return ""

    return row.get("segment_name", "")


# ============================================================
# RETRY LOGGING
# ============================================================

RETRY_LOG = OUTPUTS / "triage_retry_log.txt"


def chat_with_retry(messages, ticket_id):
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return chat(messages)

        except TransientLLMError:
            if retry_number == MAX_RETRIES:
                raise

            wait_seconds = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            log_line = (
                f"Ticket {ticket_id} | retry {retry_number} | "
                f"wait {wait_seconds:.4f} seconds"
            )

            print(log_line)

            with open(RETRY_LOG, "a", encoding="utf-8") as log:
                log.write(log_line + "\n")

            time.sleep(wait_seconds)


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_response(content):
    # First: accept valid JSON only when category is allowed.
    try:
        parsed = json.loads(content)

        category = str(parsed.get("category", "")).lower()

        if category in CATEGORIES:
            return category, "classified_json"

        # Valid JSON with an unsupported category is invalid.
        return "", "flagged_invalid_output"

    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # Second: extract lowercase words from a non-JSON response.
    words = re.findall(r"[a-z]+", str(content).lower())

    found = [category for category in CATEGORIES if category in words]

    if len(found) == 1:
        return found[0], "classified_keyword"

    return "", "flagged_invalid_output"


# ============================================================
# ROUTING RULES
# ============================================================

def route_result(ticket, status, category):
    customer_id = ticket["customer_id"]
    premium = is_premium(customer_id)

    if status in {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }:
        return "NORMAL", "human_review"

    if category == "fraud":
        return "HIGH", "security_team"

    if premium:
        return "HIGH", "premium_desk"

    return "NORMAL", "support_queue"


# ============================================================
# RUN THE PIPELINE
# ============================================================

def run_pipeline(system_prompt, token_budget=None):
    # Required: reset immediately before every independent full run.
    reset()

    total_tokens = 0
    results = []

    for ticket in prepared_tickets:
        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]
        original_text = ticket["original_text"]
        safe_text = ticket["safe_text"]

        category = ""
        status = ""
        response_tokens = 0

        # Guardrail 1: empty ticket
        if not original_text.strip():
            status = "skipped_empty"

        # Guardrail 2: input token limit
        elif est_tokens(original_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"

        # Budget check before a model call
        elif token_budget is not None and total_tokens >= token_budget:
            status = "budget_exhausted"

        else:
            # User content must be exactly the redacted ticket text.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": safe_text},
            ]

            prompt_to_model = messages[-1]["content"]

            # Privacy assertions before every call
            assert prompt_to_model == safe_text
            assert not EMAIL_PATTERN.search(prompt_to_model)
            assert not PHONE_PATTERN.search(prompt_to_model)
            assert not ORDER_PATTERN.search(prompt_to_model)

            response = chat_with_retry(messages, ticket_id)

            response_tokens = response.usage.total_tokens
            total_tokens += response_tokens

            content = response.choices[0].message.content

            category, status = validate_response(content)

        priority, route = route_result(ticket, status, category)

        results.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name(customer_id),
            "priority": priority,
            "route": route,
            "tokens_used": total_tokens,
        })

    return results, total_tokens


# ============================================================
# ACCURACY AND CONFUSION TABLE
# ============================================================

def accuracy_report(results, run_name):
    classified = [
        row for row in results
        if row["status"] in {"classified_json", "classified_keyword"}
    ]

    ticket_lookup = {
        row["ticket_id"]: row["true_category"].strip().lower()
        for row in tickets
    }

    correct = sum(
        row["category"] == ticket_lookup[row["ticket_id"]]
        for row in classified
    )

    accuracy = (correct / len(classified) * 100) if classified else 0

    print(f"\n{run_name} results")
    print("Tickets classified:", len(classified))
    print(f"Accuracy: {accuracy:.1f}%")
    print("Correct:", correct)
    print("Incorrect:", len(classified) - correct)

    return classified, accuracy


def print_confusion_table(classified):
    ticket_lookup = {
        row["ticket_id"]: row["true_category"].strip().lower()
        for row in tickets
    }

    matrix = {
        actual: {predicted: 0 for predicted in CATEGORIES}
        for actual in CATEGORIES
    }

    for row in classified:
        actual = ticket_lookup[row["ticket_id"]]
        predicted = row["category"]

        if actual in CATEGORIES and predicted in CATEGORIES:
            matrix[actual][predicted] += 1

    print("\nZero-shot confusion table")
    print("Actual \\ Predicted | " + " | ".join(CATEGORIES))

    for actual in CATEGORIES:
        values = [str(matrix[actual][predicted]) for predicted in CATEGORIES]
        print(actual + " | " + " | ".join(values))


def print_status_counts(results, label):
    counts = Counter(row["status"] for row in results)

    print(f"\n{label} status counts:")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")


# ============================================================
# ZERO-SHOT AND FEW-SHOT COMPARISON
# No token budget for these runs
# ============================================================

zero_results, zero_tokens = run_pipeline(
    ZERO_SHOT_SYSTEM,
    token_budget=None,
)

few_results, few_tokens = run_pipeline(
    FEW_SHOT_SYSTEM,
    token_budget=None,
)

zero_classified, zero_accuracy = accuracy_report(
    zero_results,
    "ZERO-SHOT",
)

few_classified, few_accuracy = accuracy_report(
    few_results,
    "FEW-SHOT",
)

print_confusion_table(zero_classified)

print("\nToken comparison")
print("Zero-shot total tokens:", zero_tokens)
print(
    "Zero-shot tokens per classified ticket:",
    round(zero_tokens / len(zero_classified), 2),
)

print("Few-shot total tokens:", few_tokens)
print(
    "Few-shot tokens per classified ticket:",
    round(few_tokens / len(few_classified), 2),
)

print_status_counts(zero_results, "Zero-shot")
print_status_counts(few_results, "Few-shot")


# ============================================================
# ZERO-SHOT BUDGET DEMO
# ============================================================

budget_results, budget_tokens = run_pipeline(
    ZERO_SHOT_SYSTEM,
    token_budget=TOKEN_BUDGET_DEMO,
)

budget_path = OUTPUTS / "triage_budget_demo.csv"

with open(budget_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used",
        ],
    )
    writer.writeheader()
    writer.writerows(budget_results)

budget_stopped = sum(
    row["status"] == "budget_exhausted"
    for row in budget_results
)

print("\nBudget demo saved:", budget_path)
print("Budget:", TOKEN_BUDGET_DEMO)
print("Tokens used:", budget_tokens)
print("Budget-exhausted tickets:", budget_stopped)


# ============================================================
# FINAL FEW-SHOT RUN
# TOKEN_BUDGET = 100000
# ============================================================

final_results, final_tokens = run_pipeline(
    FEW_SHOT_SYSTEM,
    token_budget=TOKEN_BUDGET_FINAL,
)

final_path = OUTPUTS / "triage_results.csv"

with open(final_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used",
        ],
    )
    writer.writeheader()
    writer.writerows(final_results)

print("\nFinal triage saved:", final_path)
print("Final rows:", len(final_results))
print("Final total tokens:", final_tokens)

print_status_counts(final_results, "Final few-shot")
