## Import the necessary modules

import json
import ollama

## Import the function from the module parse_data
from parse_data import load_items, get_unclaimed_items, save_result


## Build your prompt based on the description the user provides 
## and the items that are available in the lost-and-found database.
## The model must follow the rules listed in the README file
## The function should return the system prompt and the user prompt.
## You may need to use json.dumps() to convert the available_items list into a JSON string.

def build_prompt(description, available_items):
    system_prompt = (
        "You are a campus lost-and-found assistant. "
        "Your job is to find possible matches between a user's lost item description "
        "and the items available in the lost-and-found database.\n\n"
        "Rules you must follow:\n"
        "- Use ONLY the given JSON data of available items.\n"
        "- Not all details of an item must match to be a possible match.\n"
        "- Return ONLY JSON, with exactly this structure:\n"
        '{\n    "matches": ["ITEM_ID"],\n    "confidence": "LOW"\n}\n'
        '- "matches" contains all possible matching item IDs.\n'
        '- "confidence" must be exactly one of: LOW, MEDIUM, HIGH.\n'
        '- If there is no match, return an empty list for "matches".\n'
        "- Do not include any explanation or extra text outside the JSON."
    )

    available_items_json = json.dumps(available_items, indent=4)
    user_prompt = (
        f"Available items in the lost-and-found database:\n{available_items_json}\n\n"
        f"User's description of the lost item: {description}\n\n"
        "Return the possible matches as JSON."
    )

    return system_prompt, user_prompt


## Logic to ask Qwen for all the possible matches based on the system prompt and user prompt.
## The function should return the response from Qwen.
def ask_qwen(system_prompt, user_prompt):
    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


## Logic to parse the response from Qwen and return the result. 
## You may need to use json.loads() to convert the response string into a suitable Python data structure.
def parse_response(response_text):
    text = response_text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


## Logic to validate the result returned by Qwen.
## It should check if the result is a dictionary, contains the keys "matches" and "confidence", and that the values are of the correct type.
## If everything is correct, then it should check if the item IDs in the "matches" list are valid IDs .
def validate_result(result, available_items):
    if not isinstance(result, dict):
        return False
    if "matches" not in result or "confidence" not in result:
        return False
    if not isinstance(result["matches"], list):
        return False
    if not isinstance(result["confidence"], str):
        return False
    if result["confidence"] not in ["LOW", "MEDIUM", "HIGH"]:
        return False

    valid_ids = {item["id"] for item in available_items}
    for match in result["matches"]:
        if match not in valid_ids:
            return False
    return True


## Logic to display the matches found by Qwen in a user-friendly format.
## It should look something like this:
""" 
CAMPUS LOST-AND-FOUND ASSISTANT
==================================================

Describe the item you lost: I lost a black bag somewhere

Searching for possible matches...

MATCH RESULT
--------------------------------------------------
Confidence: MEDIUM

Possible matches:

ID: F101
Item: backpack
Color: black
Location: Library 2nd floor
Date found: 2026-09-15

Result saved to output/match_result.json
 """
## If no matches are found, it should display a message indicating that no matches were found, along with the empty list
def display_matches(result, available_items):
    print("\nMATCH RESULT")
    print("-" * 50)
    print(f"Confidence: {result['confidence']}")
    print()

    if not result["matches"]:
        print("No matches were found.")
        print(f"Matches: {result['matches']}")
        return

    print("Possible matches:")
    print()
    items_by_id = {item["id"]: item for item in available_items}
    for match_id in result["matches"]:
        item = items_by_id.get(match_id)
        if item:
            print(f"ID: {item['id']}")
            print(f"Item: {item['item']}")
            print(f"Color: {item['color']}")
            print(f"Location: {item['location']}")
            print(f"Date found: {item['date']}")
            print()


## Control center for the entire program.
def main():
    print("CAMPUS LOST-AND-FOUND ASSISTANT")
    print("=" * 50)
    print()

    items = load_items("found_items.json")
    available_items = get_unclaimed_items(items)

    description = input("Describe the item you lost: ")

    print("\nSearching for possible matches...")

    system_prompt, user_prompt = build_prompt(description, available_items)
    response_text = ask_qwen(system_prompt, user_prompt)

    try:
        result = parse_response(response_text)
    except json.JSONDecodeError:
        print("Error: The model did not return valid JSON.")
        return

    if not validate_result(result, available_items):
        print("Error: The model returned an invalid result.")
        print(f"Raw result: {result}")
        return

    display_matches(result, available_items)

    save_result(result, "output/match_result.json")
    print("Result saved to output/match_result.json")


if __name__ == "__main__":
    main()