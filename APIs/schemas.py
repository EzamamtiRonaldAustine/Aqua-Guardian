# Basic validation.
REQUIRED_FIELDS = [
    "timestamp",
    "temperature",
    "ph",
    "ammonia",
    "nitrate",
    "turbidity"
]


def validate_payload(payload):

    if not isinstance(payload, list):
        return False, "Payload must be a list of records."

    for record in payload:
        for field in REQUIRED_FIELDS:
            if field not in record:
                return False, f"Missing field: {field}"

    return True, None
