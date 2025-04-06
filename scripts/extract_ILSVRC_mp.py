import os
import time
import tarfile
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def extract_tar(tar_path, output_dir):
    """
    Extracts a tar file to a specified directory.
    """
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=output_dir, filter="data")

def process_category_tar(file_path, output_train_dir):
    """
    Extracts a category tar file to a specific directory and deletes the tar file.
    """
    category_name = os.path.basename(file_path).replace(".tar", "")
    category_dir = os.path.join(output_train_dir, category_name)
    os.makedirs(category_dir, exist_ok=True)
    with tarfile.open(file_path, "r") as tar:
        tar.extractall(path=category_dir, filter="data")
    os.remove(file_path)

def process_training_set(tar_file, output_dir, num_threads):
    """
    Extracts the training set tar file and organizes the dataset.
    """
    temp_dir = os.path.join(output_dir, "train_temp")
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Extracting training tar file to {temp_dir} ...")
        extract_tar(tar_file, temp_dir)
    else:
        print(f"Found existing directory {temp_dir}. Skipping extraction.")
    
    tar_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".tar")]
    print(f"Found {len(tar_files)} category tar files.")
    
    print("Sample category tar files:")
    for i, tar_file in enumerate(tar_files):
        print(f"{i+1}: {tar_file}")
        if i == 10:
            break
    
    print(f"Extracting category tar files using {num_threads} threads ...")
    output_train_dir = os.path.join(output_dir, "train")
    os.makedirs(output_train_dir, exist_ok=True)
    
    # Use ThreadPoolExecutor and pass arguments explicitly
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(lambda file: process_category_tar(file, output_train_dir), tar_files)


    
    print("Training set extraction complete.")
    print(f"Removing temporary directory {temp_dir} ...")
    os.rmdir(temp_dir)

def process_validation_set(tar_file, output_dir, valprep_script):
    """
    Extracts the validation set tar file and organizes the dataset.
    Download the valprep.sh script from here: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
    """
    val_dir = os.path.join(output_dir, "val")

    if not os.path.exists(val_dir):
        os.makedirs(val_dir, exist_ok=True)
        print(f"Extracting validation tar file to {val_dir}...")
        extract_tar(tar_file, val_dir)
    else:
        print(f"Found existing directory {val_dir}. Skipping extraction.")
    

    # Resolve the path to the valprep.sh script
    valprep_script = str(Path(valprep_script).resolve())
    print(f"Using valprep.sh script from {valprep_script}...")
    print("Organizing validation images into subdirectories...")

    if len(list(Path(val_dir).rglob("*.JPEG"))) > 0:
        subprocess.run(["bash", valprep_script], cwd=val_dir, check=True)

    
    print("Validation set extraction complete.")

def main():
    parser = argparse.ArgumentParser(description="Extract ImageNet dataset with multi-core processing.")
    parser.add_argument("--train_tar", help="Path to the training tar file.")
    parser.add_argument("--val_tar", help="Path to the validation tar file.")
    parser.add_argument("--output_dir", help="Path to the output directory.")
    parser.add_argument("--valprep_script", default="Complete path to the valprep.sh file.", help="Path to the valprep.sh script.")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of cores to use (default: 4).")
    parser.add_argument("--run_type", choices=["train", "val", "all"], default="all", help="Type of extraction to perform (default: all).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    if args.run_type in ["train", "all"]:
        if not args.train_tar:
            raise ValueError("Training tar file is required when run_type is 'train' or 'all'.")
        process_training_set(args.train_tar, args.output_dir, args.num_threads)  
    
    if args.run_type in ["val", "all"]:
        if not args.val_tar:
            raise ValueError("Validation tar file is required when run_type is 'val' or 'all'.")
        process_validation_set(args.val_tar, args.output_dir, args.valprep_script)
    end_time = time.time()
    
    print(f"Total time taken for running {args.run_type} extraction: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()

