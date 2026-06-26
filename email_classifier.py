import os
import json
from openai import OpenAI

DEFAULT_PROMPT = """You are a customer support email classifier.

Analyze the email below and respond with ONLY a JSON object in this exact format:
{{
  "category": "<one of: billing, technical, account, general>",
  "summary": "<one sentence describing the customer's issue>"
}}

Rules:
- category must be exactly one of the four options
- summary must be a single sentence, under 20 words
- Do not include any text outside the JSON object

Email to classify:
{email}"""

VALID_CATEGORIES = {"billing", "technical", "account", "general"}

def classify_support_email(
    email: str,
    prompt_template: str = DEFAULT_PROMPT,
    model: str = "gpt-4o-mini",  # fast + cheap, good for classification
) -> dict:
    """
    Classify a customer support email into a category and generate a summary.

    Args:
        email: The raw email text to classify.
        prompt_template: A prompt string with a {email} placeholder.
        model: The OpenAI model to use.

    Returns:
        {"category": str, "summary": str}

    Raises:
        ValueError: If the response can't be parsed or category is invalid.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompt = prompt_template.format(email=email)

    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Model returned non-JSON output: {raw!r}")

    if result.get("category") not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category: {result.get('category')!r}")

    if not isinstance(result.get("summary"), str):
        raise ValueError("Missing or invalid summary field")

    return {
        "category": result["category"],
        "summary": result["summary"],
    }