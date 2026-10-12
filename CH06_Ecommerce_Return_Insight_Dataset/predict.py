"""
E-Commerce Return Insight
predict.py — Terminal-based interactive prediction tool.

Usage:
    python predict.py
"""

import os
import sys
import pickle
import numpy as np

# ─────────────────────────────────────────────
# LOAD MODEL BUNDLE
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("\n  [ERROR] model.pkl not found.")
        print("  Please run:  python train_model.py  first.\n")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ─────────────────────────────────────────────
# INPUT HELPERS
# ─────────────────────────────────────────────
def ask_float(prompt, min_val=None, max_val=None):
    """Prompt for a float with optional range validation."""
    while True:
        raw = input(f"  {prompt}: ").strip()
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"    Value must be >= {min_val}. Try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"    Value must be <= {max_val}. Try again.")
                continue
            return val
        except ValueError:
            print("    Invalid input. Please enter a number.")

def ask_int(prompt, min_val=None, max_val=None):
    """Prompt for an integer with optional range validation."""
    while True:
        raw = input(f"  {prompt}: ").strip()
        try:
            val = int(raw)
            if min_val is not None and val < min_val:
                print(f"    Value must be >= {min_val}. Try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"    Value must be <= {max_val}. Try again.")
                continue
            return val
        except ValueError:
            print("    Invalid input. Please enter a whole number.")

def ask_choice(prompt, choices):
    """Display a numbered menu and return the chosen value."""
    print(f"\n  {prompt}")
    for i, c in enumerate(choices, 1):
        print(f"    [{i}] {c}")
    while True:
        raw = input("  Enter number: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
            print(f"    Please enter a number between 1 and {len(choices)}.")
        except ValueError:
            print("    Invalid input. Please enter a number.")

# ─────────────────────────────────────────────
# OPTION 1 — PREDICT RETURN FOR NEW ORDER
# ─────────────────────────────────────────────
def predict_return(bundle):
    model          = bundle["model"]
    label_encoders = bundle["label_encoders"]
    categorical_cols = bundle["categorical_cols"]
    numeric_cols   = bundle["numeric_cols"]
    encoded_cols   = bundle["encoded_cols"]
    cat_values     = bundle["category_values"]
    num_stats      = bundle["numeric_stats"]

    print("\n" + "─" * 55)
    print("  Predict Return for a New Order")
    print("─" * 55)
    print("  Fill in the order details below.")
    print("  (Press Ctrl+C at any time to go back to menu)\n")

    data = {}

    try:
        # ── Numeric inputs ──────────────────────────────
        if "quantity" in numeric_cols:
            data["quantity"] = ask_int(
                "Quantity ordered (1-20)", min_val=1, max_val=20)

        if "price" in numeric_cols:
            data["price"] = ask_float(
                "Unit price (e.g. 999.99)", min_val=0.01)

        if "discount" in numeric_cols:
            data["discount"] = ask_float(
                "Discount percentage (0-100)", min_val=0, max_val=100)

        if "delivery_delay_days" in numeric_cols:
            data["delivery_delay_days"] = ask_int(
                "Delivery delay in days\n"
                "  (negative = arrived early, 0 = on time, positive = late)",
                min_val=-30, max_val=60)

        if "order_month" in numeric_cols:
            data["order_month"] = ask_int(
                "Order month (1=Jan … 12=Dec)", min_val=1, max_val=12)

        if "order_dayofweek" in numeric_cols:
            data["order_dayofweek"] = ask_int(
                "Order day of week (0=Mon, 1=Tue, … 6=Sun)",
                min_val=0, max_val=6)

        if "order_history" in numeric_cols:
            data["order_history"] = ask_int(
                "Customer's total past orders", min_val=0, max_val=200)

        # ── Derived numeric ──────────────────────────────
        if "order_value" in numeric_cols:
            q = data.get("quantity", 1)
            p = data.get("price", 0)
            d = data.get("discount", 0)
            data["order_value"] = q * p * (1 - d / 100)

        # ── Categorical inputs ───────────────────────────
        for col in categorical_cols:
            choices = cat_values.get(col, [])
            if not choices:
                data[col] = "Unknown"
                continue
            label = col.replace("_", " ").title()
            data[col] = ask_choice(f"Select {label}:", choices)

    except KeyboardInterrupt:
        print("\n\n  Cancelled. Returning to main menu.")
        return

    # ── Build feature row ────────────────────────────────
    row = {}
    for col in numeric_cols:
        row[col] = data.get(col, 0.0)

    for col in categorical_cols:
        le = label_encoders[col]
        val = data.get(col, "Unknown")
        try:
            row[col + "_enc"] = int(le.transform([val])[0])
        except ValueError:
            # Unseen label → use 0 (most common fallback)
            row[col + "_enc"] = 0

    X_input = np.array([[row[c] for c in encoded_cols]])

    # ── Predict ──────────────────────────────────────────
    prediction  = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0]
    prob_return  = probability[1] * 100
    prob_no_ret  = probability[0] * 100

    print("\n" + "═" * 55)
    if prediction == 1:
        print("  ⚠  PREDICTION:  Return Expected")
    else:
        print("  ✔  PREDICTION:  Return Not Expected")

    print(f"\n  Probability of Return     : {prob_return:.1f}%")
    print(f"  Probability of No Return  : {prob_no_ret:.1f}%")

    # Confidence label
    conf = max(prob_return, prob_no_ret)
    if conf >= 80:
        conf_label = "High confidence"
    elif conf >= 60:
        conf_label = "Moderate confidence"
    else:
        conf_label = "Low confidence — prediction uncertain"
    print(f"  Confidence                : {conf_label}")

    print("\n  ⓘ  This is a model prediction, not a guarantee.")
    print("     Actual outcomes may differ based on factors")
    print("     not captured in the training data.")
    print("═" * 55)
    input("\n  Press Enter to return to the menu...")

# ─────────────────────────────────────────────
# OPTION 2 — SHOW MODEL PERFORMANCE
# ─────────────────────────────────────────────
def show_performance(bundle):
    metrics = bundle["metrics"]
    cm      = metrics["confusion_matrix"]

    print("\n" + "─" * 55)
    print("  Model Performance — Random Forest Classifier")
    print("─" * 55)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  "
          f"({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")

    print("\n  Confusion Matrix:")
    print("                   Predicted")
    print("                 No      Yes")
    print(f"  Actual  No   [{cm[0][0]:>5}   {cm[0][1]:>5}]")
    print(f"  Actual  Yes  [{cm[1][0]:>5}   {cm[1][1]:>5}]")

    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total = tn + fp + fn + tp
    print(f"\n  True Negatives  (correctly predicted No Return): {tn}")
    print(f"  False Positives (predicted Return, was No)     : {fp}")
    print(f"  False Negatives (predicted No, was Return)     : {fn}")
    print(f"  True Positives  (correctly predicted Return)   : {tp}")
    print(f"\n  Test set size: {total} orders")

    print("\n  Metric Definitions:")
    print("  • Accuracy  — % of all predictions that were correct")
    print("  • Precision — of predicted returns, how many were real returns")
    print("  • Recall    — of actual returns, how many were caught")
    print("  • F1 Score  — harmonic mean of precision and recall")
    print("─" * 55)
    input("\n  Press Enter to return to the menu...")

# ─────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────
def main_menu():
    bundle = load_model()

    while True:
        print("\n" + "=" * 55)
        print("     E-Commerce Return Insight")
        print("     Powered by Random Forest Classifier")
        print("=" * 55)
        print("  [1] Predict return for a new order")
        print("  [2] Show model performance")
        print("  [3] Exit")
        print("─" * 55)
        choice = input("  Enter your choice (1/2/3): ").strip()

        if choice == "1":
            predict_return(bundle)
        elif choice == "2":
            show_performance(bundle)
        elif choice == "3":
            print("\n  Goodbye!\n")
            sys.exit(0)
        else:
            print("  Invalid choice. Please enter 1, 2, or 3.")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Goodbye!\n")
        sys.exit(0)
