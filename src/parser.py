def detect_task_type(prompt: str) -> str:
    p = prompt.lower()

    if "8-bit binary" in p or "bit manipulation" in p:
        return "bit"

    if "secret encryption" in p or "decrypt" in p:
        return "cipher"

    if "unit conversion" in p:
        return "unit"

    if "gravitational constant" in p:
        return "gravity"

    if "wonderland numeral system" in p:
        return "roman"

    if "transformation rules is applied to equations" in p:
        return "symbol"

    return "unknown"