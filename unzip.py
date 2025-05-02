import zipfile, os, argparse


def unzip_large_file(zip_filepath, dest_dir):
    """
    Unzips a large zip file to a specified destination directory.

    Args:
        zip_filepath (str): The path to the large zip file.
        dest_dir (str): The directory where the contents should be extracted.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    try:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            print(f"Extracting {zip_filepath} to {dest_dir}...", end="")
            zip_ref.extractall(dest_dir)
            print("done!")
    except zipfile.BadZipFile:
        print(f"Error: {zip_filepath} is not a valid zip file.")
    except Exception as e:
        print(f"An error occurred during extraction: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str)
    parser.add_argument("--output", "-o", type=str)
    args = parser.parse_args()

    unzip_large_file(args.input, args.output)
