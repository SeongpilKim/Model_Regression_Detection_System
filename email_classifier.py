import os
import yaml
from openai import OpenAI
from schemas import EmailClassification


def classify_support_email(email: str, prompt_file: str = "prompts/email_classifier_v1.0.yaml") -> EmailClassification:
    """
    Classify a support email with a YAML-defined prompt and return the parsed result.

    The prompt file provides the system prompt, few-shot examples, and the user
    prompt template used to build the chat completion request. The model response
    is parsed into an ``EmailClassification`` instance before being returned.

    Args:
        email: The email text to classify.
        prompt_file: Path to the YAML prompt configuration file.

    Returns:
        An ``EmailClassification`` parsed from the model response.

    Raises:
        FileNotFoundError: If the prompt file cannot be opened.
        KeyError: If required keys are missing from the YAML prompt data.
        ValueError: If the model response cannot be parsed into the schema.
    """
    
    with open(prompt_file, "r") as file:
        prompt_data = yaml.safe_load(file)

    messages = []
    
    messages.append({
        "role": "system", 
        "content": prompt_data["system_prompt"]
    })
    
    # add few shot examples from prompt file
    for example in prompt_data["few_shot_examples"]:
        messages.append({
            "role": "user", 
            "content": f"Email to classify:\n{example['input']}"
        })
        messages.append({
            "role": "assistant", 
            "content": example['expected_output']
        })

    # add the actual email to classify
    final_prompt = prompt_data["user_prompt_template"].format(email=email)
    messages.append({
        "role": "user", 
        "content": final_prompt
    })


    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    response = client.beta.chat.completions.parse(
        model=prompt_data["metadata"]["target_model"],
        temperature=prompt_data["metadata"]["temperature"], 
        max_tokens=256,
        response_format=EmailClassification, 
        messages=messages,
    )
    
    parsed_result = response.choices[0].message.parsed

    # if parsed_result is None: Loudly fail
    if parsed_result is None:
        raise ValueError("Failed to parse the model's response into EmailClassification. Output may have been blocked or cut off.")
    
    # safe to return
    return parsed_result