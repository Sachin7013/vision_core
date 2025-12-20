"""
RULES - FILTERS FOR DETECTIONS
================================
Rules are conditions that decide which detected objects matter.

Think of it like:
If you detect a PERSON, keep it.
If you detect a CAR, throw it away.
"""

from typing import Any, Dict, List, Tuple
from app.models import AgentRule


def class_presence(
    rule: AgentRule,
    detections: List[Dict[str, Any]],
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    RULE: "Is this object present?"
    
    Checks if ANY object of the target class was detected.
    
    Args:
        rule: The rule with target_class (e.g., "person")
        detections: List of detected objects
    
    Returns:
        (was_matched, filtered_detections)
        - was_matched: True if we found at least 1 target object
        - filtered_detections: Only objects matching target class
    
    EXAMPLE:
    If rule says "look for PERSON"
    And we detect [PERSON, CAR, PERSON, DOG]
    Result: (True, [PERSON, PERSON])
    """
    target = rule.target_class
    
    # Filter: keep only objects matching target class
    filtered = [d for d in detections if d.get("class_name") == target]
    
    # Did we find any?
    matched = len(filtered) > 0
    
    print(f"✓ Rule '{rule.label}': Filtered {len(detections)} detections → {len(filtered)} matches")
    print(f"✓ Rule '{rule.label}': all detected object => {detections}")
    
    return (matched, filtered)


def class_count(
    rule: AgentRule,
    detections: List[Dict[str, Any]],
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    RULE: "How many objects?"
    
    Checks if we detected AT LEAST min_count of target class.
    
    Args:
        rule: The rule with target_class and min_count
        detections: List of detected objects
    
    Returns:
        (was_matched, filtered_detections)
        - was_matched: True if count >= min_count
        - filtered_detections: Only objects matching target class
    
    EXAMPLE:
    If rule says "find at least 2 PEOPLE"
    And we detect [PERSON, CAR, PERSON, DOG]
    Result: (True, [PERSON, PERSON])
    Because we have 2 people, which is >= 2
    
    If we only detected [PERSON, CAR, DOG]
    Result: (False, [PERSON])
    Because we only have 1 person, which is < 2
    """
    target = rule.target_class
    
    # Filter: keep only target class
    filtered = [d for d in detections if d.get("class_name") == target]
    
    # How many did we find?
    count = len(filtered)
    
    # What's the minimum needed? (default 1)
    min_count = rule.min_count or 1
    
    # Did we meet the minimum?
    matched = count >= min_count
    
    print(f"✓ Rule '{rule.label}': Need {min_count} {target}, found {count}")
    print(f"✓ Rule '{rule.label}': all detected object => {detections}")
    
    return (matched, filtered)