import argparse
import subprocess
import shutil
import os

# Function to download a file using wget
def download_file(url, output_file):
    try:
        subprocess.run(["wget", url, "-O", output_file], check=True)
        print(f"Downloaded {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")

# Function to extract a file using 7z
def extract_file(archive_file, output_dir="."):
    try:
        subprocess.run(["tar", "-xvzf", archive_file], check=True)
        print(f"Extracted {archive_file} to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting file: {e}")

# Function to remove a file
def remove_file(file_path):
    try:
        os.remove(file_path)
        print(f"Removed file: {file_path}")
    except OSError as e:
        print(f"Error removing file: {e}")

def main():
    output_file = "monitor.tar.gz"
    download_file("https://zenodo.org/records/14927551/files/monitor.tar?download=1", output_file)
    extract_file(output_file)
    remove_file(output_file)

if __name__ == "__main__":
    main()