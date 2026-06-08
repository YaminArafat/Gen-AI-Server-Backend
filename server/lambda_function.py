import json
from pydantic import ValidationError
from server.schemas import Config

def lambda_handler(event, context):

    try:
        body = event.get("body", "{}")
        if isinstance(body, str):
            body = json.loads(body)
            
        raw_json_output = body.get("raw_model_output")
        
        validated_config = Config.model_validate_json(raw_json_output)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "valid": True,
                "config": validated_config.model_dump()
            })
        }
    except ValidationError as val_err:
        return {
            "statusCode": 422,
            "body": json.dumps({
                "valid": False,
                "error": "Pydantic Schema Alignment Failure",
                "details": val_err.errors()
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"valid": False, "error": str(e)})
        }