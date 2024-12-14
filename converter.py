import pandas as pd
import json

# Load the JSON data
with open('data.json', 'r') as file:
    data = json.load(file)

# Convert the JSON data to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('data.csv', index=False)

print("JSON data has been converted to data.csv")