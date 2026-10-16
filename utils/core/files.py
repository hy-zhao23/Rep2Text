import os
import json
import pickle
from pathlib import Path
import time
from .logger import logger

def get_layer_num(layer: str):
    layer = layer.split('_')[0]
    return int(''.join(c for c in str(layer) if c.isdigit()))

def remove_files(files):
    for file in files:
        file.unlink()

def wait_for_files(files):
    '''
    Wait until all checkpoint files exist
    '''
    try:
        while True:
            all_exist = True
            for file in files:
                if not file.exists():
                    all_exist = False
                    break
            if all_exist:
                return
            time.sleep(300)
            logger.info(f"⏳ Waiting for checkpoint files to complete...")

    except Exception as e:
        logger.error(f"❌ Error in checkpoint_files_exist: {e}")
        raise e

def write_pkl(data, filename, verbose=True):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        tmp_filename = str(filename) + ".tmp"
        with open(tmp_filename, 'wb') as f:
            pickle.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_filename, filename)  # atomic on most systems
        if verbose:
            logger.info(f"✅ Data successfully written to {filename}")
        return True
    except Exception as e:
        logger.error(f"❌ Unexpected error while writing to {filename}: {e}")
        raise e

def read_pkl(filename):
    try:
        if Path(filename).stat().st_size == 0:
            logger.warning(f"❗ {filename} is empty.")
            return None
        
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.warning(f"❗ File {filename} not found.")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error while reading {filename}: {e}")
        raise e

def read_json(filename):
    try:
        if Path(filename).stat().st_size == 0:
            logger.warning(f"{filename} is empty.")
            return None
        
        
        with open(filename, 'r', encoding='utf-8') as f:
            if filename.lower().endswith('.jsonl'):
                # Read JSONL file line by line
                return [json.loads(line.strip()) for line in f if line.strip()]
            else:
                # Read regular JSON file
                return json.load(f)
    except FileNotFoundError:
        logger.warning(f"❗ File {filename} does not exist.")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error while reading {filename}: {e}")
        raise e
    
def write_json(data, filename):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        tmp_filename = str(filename) + ".tmp"
        with open(tmp_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_filename, filename)  # atomic on most systems
        logger.info(f"✅ Generated concepts have been successfully written to {filename}")
        return True
    except Exception as e:
        logger.error(f"❌ Unexpected error while writing to {filename}: {e}")
        raise e


def read_file(file_path: str):
    try:
        with open(file_path, "r") as f:
            return f.readlines()
    except Exception as e:
        logger.error(f"❌ Error in read_files: {e}")
        raise e
    
def write_txt(data, filename):
    '''
    Write data to a txt file
    '''
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        tmp_filename = str(filename) + ".tmp"
        with open(tmp_filename, 'w') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_filename, filename)  # atomic on most systems
        logger.info(f"✅ Data successfully written to {filename}")
        return True
    except Exception as e:
        logger.error(f"❌ Error in write_txt: {e}")
        raise e
