
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"

CLEAN.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Load raw tickets
tickets = pd.read_csv(
    RAW / "tickets.csv",
    keep_default_na=False
)

print("Original ticket count:", len(tickets))
print("Columns:", tickets.columns.tolist())

# Remove duplicate ticket IDs
tickets = tickets.drop_duplicates(subset="ticket_id").copy()

# Clean text fields
for column in ["ticket_id", "customer_id", "order_id", "text", "true_category"]:
    tickets[column] = tickets[column].astype(str).str.strip()

# Mask email addresses and phone numbers
def anonymize_text(text):
    text = str(text)

    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        text
    )

    text = re.sub(
        r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d[-.\s]?){10}\b",
        "[PHONE]",
        text
    )

    return text

tickets["text"] = tickets["text"].apply(anonymize_text)

# Remove rows with missing essential fields
tickets = tickets[
    (tickets["ticket_id"] != "") &
    (tickets["customer_id"] != "") &
    (tickets["text"] != "") &
    (tickets["true_category"] != "")
].copy()

# Standardize category labels
tickets["true_category"] = (
    tickets["true_category"].str.lower().str.strip()
)

print("\nTicket category counts:")
print(tickets["true_category"].value_counts())

# Save cleaned tickets
output_path = CLEAN / "tickets_clean.csv"
tickets.to_csv(output_path, index=False)

print("\nTicket cleaning completed!")
print("Tickets remaining:", len(tickets))
print("Saved:", output_path)

# ==========================================
# PART C - Ticket Classification and Routing
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned tickets
tickets = pd.read_csv(CLEAN / "tickets_clean.csv")

# Load customer segments created in Part B
segments = pd.read_csv(OUTPUTS / "customer_segments.csv")

# Create training and testing data
X = tickets["text"]
y = tickets["true_category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build ticket classification pipeline
ticket_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# Train model
ticket_model.fit(X_train, y_train)

# Evaluate model
test_predictions = ticket_model.predict(X_test)

print("\nTicket Classification Results:")
print("Accuracy:", round(
    accuracy_score(y_test, test_predictions), 3
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

# Predict category for every ticket
tickets["predicted_category"] = ticket_model.predict(
    tickets["text"]
)

# Join customer segment information
tickets = tickets.merge(
    segments[["customer_id", "segment_name", "is_premium"]],
    on="customer_id",
    how="left"
)

# Route ticket to the correct support team
routing_teams = {
    "fraud": "Fraud Investigation Team",
    "billing": "Billing Support Team",
    "delivery": "Delivery Support Team",
    "technical": "Technical Support Team"
}

tickets["assigned_team"] = tickets["predicted_category"].map(
    routing_teams
)

# Priority based on ticket category and customer segment
def assign_priority(row):
    if row["predicted_category"] == "fraud":
        return "Critical"

    if row["is_premium"] == True:
        return "High"

    if row["predicted_category"] in ["billing", "delivery"]:
        return "Medium"

    return "Normal"

tickets["priority"] = tickets.apply(
    assign_priority,
    axis=1
)

# Save routed tickets
routing_output = tickets[
    [
        "ticket_id",
        "customer_id",
        "predicted_category",
        "segment_name",
        "is_premium",
        "assigned_team",
        "priority"
    ]
]

routing_output.to_csv(
    OUTPUTS / "ticket_routing.csv",
    index=False
)

print("\nTicket Routing Summary:")
print(routing_output["assigned_team"].value_counts())

print("\nPriority Summary:")
print(routing_output["priority"].value_counts())

print("\nTicket classification and routing completed!")
print("Saved:", OUTPUTS / "ticket_routing.csv")

