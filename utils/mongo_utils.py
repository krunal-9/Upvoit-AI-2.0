from datetime import datetime,date
from decimal import Decimal
from bson import Decimal128,ObjectId

def make_mongo_safe(obj):
    if isinstance(obj, dict):
        return {k: make_mongo_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_mongo_safe(v) for v in obj]
    if isinstance(obj, Decimal):
        return Decimal128(obj)
    if isinstance(obj, datetime):
        return obj   # pymongo handles datetime automatically
    return obj


def serialize_mongo_value(value):
    """
    Safely convert MongoDB/BSON types to JSON-serializable values.
    """
    if isinstance(value, ObjectId):
        return str(value)

    if isinstance(value, Decimal128):
        # Convert to float OR string (string is safer for precision)
        return float(value.to_decimal())
        # or: return str(value.to_decimal())

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat() + "Z"

    if isinstance(value, dict):
        return {k: serialize_mongo_value(v) for k, v in value.items()}

    if isinstance(value, list):
        return [serialize_mongo_value(v) for v in value]

    return value
