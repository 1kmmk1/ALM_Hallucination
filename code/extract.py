import os
import json
import time
import argparse
from typing import List, Tuple

from openai import OpenAI
from tqdm import tqdm


def extract_audio_facts(caption: str, client: OpenAI) -> List[Tuple[str, str]]:
    """Extract (object, attribute) pairs from audio caption using GPT-4."""
    
    prompt = f"""Given an audio caption, extract atomic facts in the format of (object, attribute) pairs.
    Task: Break down the audio caption into independent sound events. For each sound event, identify:
    - Object: The source of the sound (person, animal, object, etc.). Use "unspecified" if not mentioned.
    - Attribute: The characteristic of the sound (action, quality, manner, etc.)
    Output format: Return ONLY a valid JSON array of [object, attribute] pairs.
    Examples:
    Caption: "A man is laughing and a woman is crying"
    Output: [["man", "laughing"], ["woman", "crying"]]
    Caption: "Laughing and crying"
    Output: [["unspecified", "laughing"], ["unspecified", "crying"]]
    Caption: "A car engine starting"
    Output: [["car engine", "starting"]]
    Caption: "Birds chirping cheerfully and wind blowing gently"
    Output: [["birds", "chirping"], ["wind", "blowing"]]
    Caption: "Music playing"
    Output: [["music", "playing"]]
    Caption: "A baby crying, a woman speaking soothingly"
    Output: [["baby", "crying"], ["woman", "speaking"]]
    Now extract facts from this caption:
    Caption: "{caption}" 
    Output: """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    response_text = response.choices[0].message.content.strip()
    
    # Remove markdown code blocks
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
    response_text = response_text.strip()
    
    try:
        facts = json.loads(response_text)
        return [(obj, attr) for obj, attr in facts]
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response: {response_text}")
        return []


def extract_with_retry(caption: str, client: OpenAI, max_retries: int = 3) -> List[Tuple[str, str]]:
    """Extract facts with retry logic."""
    for attempt in range(max_retries):
        try:
            return extract_audio_facts(caption, client)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    return []


def process_results(input_path: str, output_path: str, api_key: str = None):
    """Process caption results and extract atomic facts."""
    client = OpenAI(api_key=api_key)
    
    checkpoint_path = f"{output_path}.checkpoint.json"
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            output = json.load(f)
        processed_count = len(output)
    else:
        output = []
        processed_count = 0
    
    # Load input data
    with open(input_path, "r") as f:
        data = json.load(f)
    
    print(f"Total: {len(data)}, Already processed: {processed_count}")
    
    for idx in tqdm(range(processed_count, len(data))):
        item = data[idx]
        
        try:
            item["original_facts"] = extract_with_retry(item["original_caption"], client)
            time.sleep(2)
            
            item["swapped_facts"] = extract_with_retry(item["swapped_caption"], client)
            time.sleep(2)
            
            output.append(item)
            
            # Save checkpoint
            with open(checkpoint_path, "w") as f:
                json.dump(output, f, indent=2)
                
        except Exception as e:
            print(f"Error at item {idx}: {e}")
            continue
    
    # Save final output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSON path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    args = parser.parse_args()
    
    process_results(args.input, args.output, args.api_key)
