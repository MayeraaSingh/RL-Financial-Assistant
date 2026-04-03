from .models import Observation, Expense

# We use static datasets for the tasks as per requirements.

TASKS = {
    "1": {
        "id": "1",
        "description": "Identify the highest expense category.",
        "observation": Observation(
            expenses=[
                Expense(category="food", amount=1200.0),
                Expense(category="rent", amount=2500.0),
                Expense(category="entertainment", amount=300.0),
            ],
            budget=4500.0,
            total_spent=4000.0
        )
    },
    "2": {
        "id": "2",
        "description": "Detect overspending against a defined budget and suggest a reduction.",
        "observation": Observation(
            expenses=[
                Expense(category="shopping", amount=2000.0),
                Expense(category="dining", amount=1500.0),
                Expense(category="utilities", amount=400.0),
            ],
            budget=3000.0,
            total_spent=3900.0
        )
    },
    "3": {
        "id": "3",
        "description": "Provide a full financial advisory summarizing patterns and giving actionable recommendations.",
        "observation": Observation(
            expenses=[
                Expense(category="subscription_netflix", amount=20.0),
                Expense(category="subscription_gym", amount=50.0),
                Expense(category="subscription_spotify", amount=15.0),
                Expense(category="coffee", amount=300.0),
                Expense(category="groceries", amount=600.0),
            ],
            budget=1500.0,
            total_spent=985.0
        )
    }
}
