#this script only copies 24 files ever 120 files
#%%
import os
import shutil

def batch_copy(src_root, dst_root, copy_batch=24, interval=120):
    """
    Copy batches of `copy_batch` files every `interval` files from src_root to dst_root,
    preserving directory structure and filenames.
    """
    # Counter for total files encountered
    file_counter = 0

    for dirpath, dirs, files in os.walk(src_root):
        # Determine the corresponding destination directory
        rel_dir = os.path.relpath(dirpath, src_root)
        dst_dir = os.path.join(dst_root, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)

        for filename in sorted(files):
            file_counter += 1
            # Determine if this file falls into a batch to copy
            position_in_cycle = (file_counter - 1) % interval
            if position_in_cycle < copy_batch:
                src_file = os.path.join(dirpath, filename)
                dst_file = os.path.join(dst_dir, filename)
                try:
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied: {src_file} -> {dst_file}")
                except Exception as e:
                    print(f"Failed to copy {src_file}: {e}")






dir = 'Z:\\Norsar01v2'

days = os.listdir(dir)
days = days[0:3] + days[4:6]

for d in days:
    data_dir = os.path.join(dir,d)
    dst_dir = os.path.join('F:\\data',d)
    batch_copy(src_root = data_dir, dst_root = dst_dir, copy_batch=24, interval=120)
    


# %%
