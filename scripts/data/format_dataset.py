import json
import argparse
import os

def format_example_to_chatml(instruction: str, thought: str, response: str, system_prompt: str = None) -> dict:
    """
    Formats a raw row into ChatML with explicit DeepSeek-R1 <think> tags.
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        
    # Add User Prompt
    messages.append({"role": "user", "content": instruction.strip()})
    
    # Format Assistant response ensuring <think> tags exist
    thought_clean = thought.strip() if thought else "Analyze the task requirements, edge cases, and design the solution."
    
    # Clean thinking tags if they already exist in raw data
    thought_clean = thought_clean.replace("<think>", "").replace("</think>", "").strip()
    response_clean = response.strip()
    
    full_assistant_content = f"<think>\n{thought_clean}\n</think>\n{response_clean}"
    
    messages.append({"role": "assistant", "content": full_assistant_content})
    
    return {"messages": messages}

def main():
    parser = argparse.ArgumentParser(description="Convert raw instruction data to ChatML JSONL for DeepSeek-R1")
    parser.add_argument("--input_file", type=str, required=True, help="Path to raw json/jsonl file")
    parser.add_argument("--output_file", type=str, default="data/processed_dataset.jsonl", help="Output ChatML JSONL path")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    formatted_count = 0
    
    with open(args.input_file, "r", encoding="utf-8") as infile:
        if args.input_file.endswith(".jsonl"):
            lines = infile.readlines()
            raw_data = [json.loads(line) for line in lines]
        else:
            raw_data = json.load(infile)

    with open(args.output_file, "w", encoding="utf-8") as outfile:
        for entry in raw_data:
            # Map flexible raw key names
            instruction = entry.get("instruction") or entry.get("prompt") or entry.get("user")
            thought = entry.get("think") or entry.get("reasoning") or entry.get("thought") or ""
            response = entry.get("response") or entry.get("output") or entry.get("assistant") or entry.get("code")

            if instruction and response:
                formatted_entry = format_example_to_chatml(
                    instruction=instruction,
                    thought=thought,
                    response=response,
                    system_prompt=args.system_prompt
                )
                outfile.write(json.dumps(formatted_entry, ensure_ascii=False) + "\n")
                formatted_count += 1

    print(f"Successfully processed {formatted_count} rows -> Saved to {args.output_file}")

if __name__ == "__main__":
    main()