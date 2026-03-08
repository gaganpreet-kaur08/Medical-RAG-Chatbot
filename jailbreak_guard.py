JAILBREAK_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard safety",
    "act as an unrestricted ai",
    "bypass rules",
    "jailbreak",
    "pretend you are not restricted",
    "do anything now",
]


def detect_jailbreak(query: str) -> bool:
    """
    Detects possible jailbreak or malicious prompts.
    Returns True if unsafe query is detected.
    """

    query_lower = query.lower()

    for pattern in JAILBREAK_PATTERNS:
        if pattern in query_lower:
            return True

    return False