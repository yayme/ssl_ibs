import os
import shutil

# create the target directory if it doesn't exist
target_dir = 'results_json_old_11082025'
os.makedirs(target_dir, exist_ok=True)

# iterate over files in the current directory
for file in os.listdir('.'):
    if file.endswith('.json') and not file.startswith('config'):
        shutil.move(file, os.path.join(target_dir, file))

print(f"All PNG files moved to '{target_dir}/'")
