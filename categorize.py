import ollama
import os

model = 'llama3.2'

input_path='./data/grocery_list.txt'
output_path='./data/grocery_list_categorized.txt'

if not os.path.exists(input_path):
    print(f'Error: {input_path} does not exist')
    exit(1)

with open(input_path, 'r') as f:
    grocery_list = f.read().split()

prompt=f"""
You are an assitant that categorizes grocery items.
Heres the list of items to categorize:
{grocery_list}

Return the response in the following format and sort items under categories and categories in alphabetical order:
Category
1. Item1
2. Item2
3. Item3
"""

try:
    response = ollama.generate  (
        model=model,
        prompt=prompt,
    )
except Exception as e:
    print(f'Error: {e}')
    exit(1)

print(response['response'])