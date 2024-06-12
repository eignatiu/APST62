import torch
import os

# Load the model
model = torch.load(r'APST62\NEON.pt')

# Specify the directory where you want to save the file
directory = r'C:\Temp\APST62'

# Create the full file path
file_path = os.path.join(directory, 'model_structure.txt')

# Open the file in write mode
with open(file_path, 'w') as f:
    # Write the model structure to the file
    print(model, file=f)

print(f"The model structure was successfully written to {file_path}")

print("The model structure was successfully written to model_structure.txt")
