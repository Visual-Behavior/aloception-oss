""" File system utils methods
"""
import os
from shutil import copyfile, move


def move_and_replace(src_dir_path: str, tgt_dir_path: str):
    """This method will copy the full `src_dir_path` into an other
    `tgt_dir_path`. If some file already exists in the target dir, this
    method will compare both file and replace the target file if the
    src file and tgt file do not match.

    Parameters
    ----------
    src_dir_path : str
        Source directory
    tgt_dir_path : str
        Target directory
    """
    # Merge the WIP directory with the target dir
    for src_dir, dirs, files in os.walk(src_dir_path):
        dst_dir = src_dir.replace(src_dir_path, tgt_dir_path, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                if os.path.samefile(src_file, dst_file):
                    continue
                os.remove(dst_file)
            move(src_file, dst_file)
