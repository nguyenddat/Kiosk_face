import os
import pickle

def initialize_folder() -> None:
    home = get_home()
    weights_path = os.path.join(home, "weights")
    
    if not os.path.exists(weights_path):
        os.makedirs(weights_path, exist_ok=True)
        print(f"Directory {weights_path} has been created")

def get_home():
    return str(os.path.join(os.getcwd()))

def load_file(file_path):
    if os.path.isfile(file_path):
        with open(file_path, "rb") as file:
            return pickle.load(file)
    
    raise FileNotFoundError(f"Không thể mở file: {file_path}")

def save_file(objs, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(objs, file)