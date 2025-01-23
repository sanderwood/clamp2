input_dir = "<path_to_your_abc_files>"  # Replace with the path to your folder containing standard ABC (.abc) files

import os
import re
import random
from multiprocessing import Pool
from tqdm import tqdm
from abctoolkit.utils import (
    find_all_abc, 
    remove_information_field, 
    remove_bar_no_annotations, 
    Quote_re, 
    Barlines, 
    strip_empty_bars
)
from abctoolkit.rotate import rotate_abc
from abctoolkit.check import check_alignment_unrotated

def abc_pipeline(abc_path, input_dir, output_dir):
    """
    Converts standard ABC notation to interleaved ABC notation.
    """
    with open(abc_path, 'r', encoding='utf-8') as f:
        abc_lines = f.readlines()

    abc_lines = [line for line in abc_lines if line.strip() != '']
    abc_lines = remove_information_field(abc_lines=abc_lines,
                                         info_fields=['X:', 'T:', 'C:', 'W:', 'w:', 'Z:', '%%MIDI'])
    abc_lines = remove_bar_no_annotations(abc_lines)

    # Remove escaped quotes and clean up barlines inside quotes
    for i, line in enumerate(abc_lines):
        if not (re.search(r'^[A-Za-z]:', line) or line.startswith('%')):
            abc_lines[i] = line.replace(r'\"', '')
            quote_contents = re.findall(Quote_re, line)
            for quote_content in quote_contents:
                for barline in Barlines:
                    if barline in quote_content:
                        line = line.replace(quote_content, '')
                        abc_lines[i] = line

    try:
        stripped_abc_lines, bar_counts = strip_empty_bars(abc_lines)
    except Exception as e:
        print(abc_path, 'Error in stripping empty bars:', e)
        return

    if stripped_abc_lines is None:
        print(abc_path, 'Failed to strip')
        return

    # Check alignment
    _, bar_no_equal_flag, bar_dur_equal_flag = check_alignment_unrotated(stripped_abc_lines)
    if not bar_no_equal_flag:
        print(abc_path, 'Unequal bar number')
    if not bar_dur_equal_flag:
        print(abc_path, 'Unequal bar duration (unaligned)')

    # Construct the output path, maintaining input folder structure
    relative_path = os.path.relpath(abc_path, input_dir)  # Get relative path from input dir
    output_file_path = os.path.join(output_dir, os.path.normpath(relative_path))  # Recreate output path
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # Ensure output folder exists

    try:
        rotated_abc_lines = rotate_abc(stripped_abc_lines)
    except Exception as e:
        print(abc_path, 'Error in rotating:', e)
        return

    if rotated_abc_lines is None:
        print(abc_path, 'Failed to rotate')
        return

    with open(output_file_path, 'w', encoding='utf-8') as w:
        w.writelines(rotated_abc_lines)


def abc_pipeline_list(abc_path_list, input_dir, output_dir):
    for abc_path in tqdm(abc_path_list):
        try:
            abc_pipeline(abc_path, input_dir, output_dir)
        except Exception as e:
            print(abc_path, e)
            pass


def batch_abc_pipeline(input_dir):
    """
    Batch process all ABC files from `input_dir`, converting them to interleaved notation.
    """
    output_dir = input_dir + "_interleaved"
    os.makedirs(output_dir, exist_ok=True)

    abc_path_list = [abc_path for abc_path in find_all_abc(input_dir) if os.path.getsize(abc_path) > 0]
    random.shuffle(abc_path_list)
    print(f"Found {len(abc_path_list)} ABC files.")

    num_cpus = os.cpu_count()
    split_lists = [abc_path_list[i::num_cpus] for i in range(num_cpus)]

    with Pool(processes=num_cpus) as pool:
        pool.starmap(abc_pipeline_list, [(split, input_dir, output_dir) for split in split_lists])


if __name__ == '__main__':
    batch_abc_pipeline(input_dir)

