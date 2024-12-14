### Modular Code for Dynamic Synthetic Data Generation

#### **File: synthetic_data_pipeline.py**
import json
import pandas as pd
from multiprocessing import Pool
from ollama import chat

# Function to call the Ollama model via Python library
def call_ollama_model(prompt, model_name="llama3.2", max_tokens=500, retries=3):
    """
    Generate text using the locally installed Ollama model.
    Retry logic is added for robustness.
    """
    for attempt in range(retries):
        try:
            response = chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
            )
            if not response or not response.message or not response.message.content:
                raise ValueError("Received an empty or invalid response from the Ollama model.")
            return response.message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise ValueError("Exceeded maximum retry attempts for Ollama model.")

# Function to refine user prompt
def refine_and_generate_template(user_prompt, model_name="llama3.2"):
    """
    Refine a vague user prompt into a structured template.
    """
    refinement_prompt = f"""
    The user provided the following prompt: "{user_prompt}"
    Extract the intent and missing details, then generate a detailed dataset generation request. Include fields, types, ranges, and other details as needed. Provide it in this JSON format:
    {{
        "intent": "<Dataset purpose>",
        "fields": [
            {{"name": "<FieldName>", "type": "<Type>", "range": "<Optional Range>"}},
            ...
        ],
        "rows": <Number of rows>
    }}
    """

    refined_response = call_ollama_model(refinement_prompt, model_name=model_name, max_tokens=300)

    if not refined_response:
        raise ValueError("Model failed to generate a valid JSON response for refinement.")

    try:
        # Extract JSON content from response
        json_start = refined_response.find('{')
        json_end = refined_response.rfind('}') + 1
        refined_template = json.loads(refined_response[json_start:json_end])
    except json.JSONDecodeError:
        print(f"Response content: {refined_response}")
        raise ValueError("Model response is not valid JSON. Ensure the prompt is detailed enough.")

    return refined_template

# Function to validate tabular data
def validate_generated_data(data, fields):
    """
    Validate the generated data against specified field constraints.
    """
    for field in fields:
        name = field['name']
        if field['type'] == 'numeric' and 'range' in field:
            min_val, max_val = map(float, field['range'].strip('[]').split(','))
            if not all(min_val <= row[name] <= max_val for row in data):
                raise ValueError(f"Field '{name}' contains values outside the range {field['range']}")
        if field['type'] == 'unique_id':
            unique_values = {row[name] for row in data}
            if len(unique_values) != len(data):
                raise ValueError(f"Field '{name}' contains duplicate values, expected unique IDs.")
    print("Validation passed for all fields.")

# Function to generate synthetic data
def generate_synthetic_data(refined_template, model_name="llama3.2"):
    """
    Use the refined template to generate synthetic data.
    """
    generation_prompt = f"""
    Generate {refined_template['rows']} rows of synthetic data in JSON format. Only return a JSON array. Fields:
    {', '.join([f"{field['name']} ({field['type']})" + (f", range: {field['range']}" if 'range' in field else "") for field in refined_template['fields']])}.
    """

    generated_data = call_ollama_model(generation_prompt, model_name=model_name, max_tokens=1000)

    if not generated_data:
        raise ValueError("Model failed to generate a valid JSON response for data.")

    try:
        # Extract only the JSON array from the response
        json_start = generated_data.find('[')
        json_end = generated_data.rfind(']') + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No valid JSON array found in the model's response.")
        data = json.loads(generated_data[json_start:json_end])
    except json.JSONDecodeError as e:
        print(f"Response content: {generated_data}")
        raise ValueError(f"Model response is not valid JSON. Error: {e}")

    # Validate data
    validate_generated_data(data, refined_template['fields'])

    return data


# Function for batch processing of large datasets
def save_batches_to_disk(prompt_template, total_rows, batch_size, model_name="llama3.2"):
    """
    Generate and save large synthetic datasets in batches to disk.
    """
    num_batches = (total_rows // batch_size) + (1 if total_rows % batch_size != 0 else 0)

    for batch_num in range(num_batches):
        prompt = prompt_template.format(batch_size=batch_size)
        response = call_ollama_model(prompt, model_name=model_name, max_tokens=500)

        if not response:
            print(f"Error in batch {batch_num}: Received no response from the model.")
            continue

        try:
            batch_data = json.loads(response)
            with open(f"dataset_batch_{batch_num}.csv", "w") as f:
                pd.DataFrame(batch_data).to_csv(f, index=False)
        except Exception as e:
            print(f"Error in batch {batch_num}: {e}")

    print("All batches saved to disk.")

# Multiprocessing for distributed generation
def process_batch(batch_num, refined_template, batch_size, model_name="llama3.2"):
    """
    Generate synthetic data for a batch using the refined template.
    """
    refined_template["rows"] = batch_size
    return generate_synthetic_data(refined_template, model_name=model_name)

def parallel_data_generation(user_prompt, total_rows, batch_size, model_name="llama3.2"):
    """
    Generate large datasets using multiprocessing.
    """
    refined_template = refine_and_generate_template(user_prompt, model_name=model_name)
    num_batches = (total_rows // batch_size) + (1 if total_rows % batch_size != 0 else 0)

    with Pool(processes=4) as pool:  # Adjust number of processes based on CPU cores
        results = pool.starmap(process_batch, [(i, refined_template, batch_size, model_name) for i in range(num_batches)])

    # Combine results from all batches
    all_data = [row for batch in results for row in batch]
    pd.DataFrame(all_data).to_csv("large_synthetic_dataset.csv", index=False)
    print("Dataset generation completed and saved to large_synthetic_dataset.csv.")

# Main Execution
def main():
    user_prompt = "Generate sales data"
    total_rows = 100000  # Adjust as needed
    batch_size = 1000    # Adjust as needed

    # Generate dataset
    parallel_data_generation(user_prompt, total_rows, batch_size, model_name="llama3.2")

if __name__ == "__main__":
    main()
