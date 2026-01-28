from datetime import datetime, date
from decimal import Decimal

def serialize_data(obj):
    """Recursively convert Decimal to float and datetime to str for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_data(i) for i in obj]
    return obj

