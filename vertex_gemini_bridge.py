import json
import sys


def main():
    payload = json.loads(sys.stdin.read())

    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=payload["project"],
        location=payload["location"],
    )
    try:
        response = client.models.generate_content(
            model=payload["model"],
            contents=payload["contents"],
            config=types.GenerateContentConfig(
                system_instruction=payload.get("system_instruction"),
                max_output_tokens=payload["max_output_tokens"],
                temperature=payload["temperature"],
            ),
        )
        usage_metadata = response.usage_metadata
        result = {
            "text": response.text,
            "usage": {
                "prompt_tokens": getattr(usage_metadata, "prompt_token_count", None),
                "completion_tokens": getattr(usage_metadata, "candidates_token_count", None),
                "total_tokens": getattr(usage_metadata, "total_token_count", None),
                "traffic_type": str(getattr(usage_metadata, "traffic_type", "")),
            },
        }
        print(json.dumps(result))
    finally:
        client.close()


if __name__ == "__main__":
    main()
