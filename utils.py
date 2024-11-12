import os
import shutil
import json


def check_folder_and_solve(output_directory, force=False):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    elif os.listdir(output_directory):
        if force:
            response = "yes"
        else:
            response = input(f"Files already exist in directory {output_directory}. "
                             f"Are you sure you want to overwrite them? (y/n): ")
        if response not in ["yes", "y", "YES", "Y"]:
            print("STOPPED!")
            exit()
        for filename in os.listdir(output_directory):
            file_path = os.path.join(output_directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
                exit()
        if not force:
            print("Output folder is now empty.\n")

