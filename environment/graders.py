import re
from .models import Action, Reward

def grade_task_1(action: Action) -> Reward:
    # Target highest category: 'rent'
    analysis_text = action.analysis.lower()
    if "rent" in analysis_text:
        return Reward(score=1.0, feedback="Correctly identified 'rent' as the highest expense.")
    return Reward(score=0.0, feedback="Failed to identify 'rent' as the highest expense.")

def grade_task_2(action: Action) -> Reward:
    # Target: Detect overspending and suggest reduction
    # Overspending indication: Total spent (3900) > Budget (3000). Total overspent is 900.
    analysis_text = action.analysis.lower()
    recommendation_text = action.recommendation.lower()
    
    score = 0.0
    feedback = []
    
    if re.search(r'(over|exceed|3900|900|deficit)', analysis_text):
        score += 0.5
        feedback.append("Successfully identified overspending context.")
    else:
        feedback.append("Did not explicitly mention overspending or budget exceeded.")
        
    if re.search(r'(reduce|cut|decrease|shopping|dining|lower)', recommendation_text):
        score += 0.5
        feedback.append("Provided a valid reduction suggestion.")
    else:
        feedback.append("Missing actionable recommendation to cut back expenses.")
        
    return Reward(score=score, feedback=" ".join(feedback))

def grade_task_3(action: Action) -> Reward:
    # Target: Full advisory (pattern detection + actionable recommendations)
    # Patterns: Multiple subscriptions, high coffee expense.
    analysis_text = action.analysis.lower()
    recommendation_text = action.recommendation.lower()
    
    score = 0.0
    feedback = []
    
    # Check pattern detection
    if "sub" in analysis_text or "multiple" in analysis_text:
        score += 0.3
        feedback.append("Detected pattern around subscriptions.")
    else:
        feedback.append("Missed subscription pattern.")
        
    if "coffee" in analysis_text:
        score += 0.2
        feedback.append("Detected high coffee expense.")
        
    # Check recommendation actionability
    if re.search(r'(cancel|consolidate|cut|brew|make|limit|stop)', recommendation_text):
        score += 0.5
        feedback.append("Gave realistic actionable recommendation based on patterns.")
    else:
        feedback.append("Recommendations lacked specific actions for patterns found.")
        
    return Reward(score=score, feedback=" ".join(feedback))

GRADERS = {
    "1": grade_task_1,
    "2": grade_task_2,
    "3": grade_task_3,
}
