import kagglehub
import os
import shutil

destination_path = os.path.join(os.getcwd(), "data/raw_data")

# Download into KaggleHub cache
cache_path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")
print("KaggleHub cache at:", cache_path)

# Copy the whole folder to your raw_data
shutil.copytree(cache_path, destination_path, dirs_exist_ok=True)

print("Dataset copied to:", destination_path)
