input_dir = "<path_to_your_xml_files>"  # Replace with the path to your folder containing XML (.xml, .mxl, .musicxml) files

import os
import math
import random
import subprocess
from tqdm import tqdm
from multiprocessing import Pool


def convert_xml2abc(file_list):
    cmd = 'python utils/xml2abc.py -d 8 -x '
    for file in tqdm(file_list):
        filename = os.path.basename(file)
        output_dir = os.path.dirname(file)
        output_dir = os.path.join(output_dir + '_abc')  # Add '_abc' to the output directory
        os.makedirs(output_dir, exist_ok=True)

        try:
            p = subprocess.Popen(cmd + '"' + file + '"', stdout=subprocess.PIPE, shell=True)
            result = p.communicate()
            output = result[0].decode('utf-8')

            if output == '':
                with open("logs/xml2abc_error_log.txt", "a", encoding="utf-8") as f:
                    f.write(file + '\n')
                continue
            else:
                with open(os.path.join(output_dir, filename.rsplit('.', 1)[0] + '.abc'), 'w', encoding='utf-8') as f:
                    f.write(output)
        except Exception as e:
            with open("logs/xml2abc_error_log.txt", "a", encoding="utf-8") as f:
                f.write(file + ' ' + str(e) + '\n')


if __name__ == '__main__':
    file_list = []
    os.makedirs("logs", exist_ok=True)

    # Traverse the specified folder for XML/MXL files
    for root, dirs, files in os.walk(os.path.abspath(input_dir)):
        for file in files:
            if file.endswith((".mxl", ".xml", ".musicxml")):
                filename = os.path.join(root, file).replace("\\", "/")
                file_list.append(filename)

    # Shuffle and prepare for multiprocessing
    random.shuffle(file_list)
    num_files = len(file_list)
    num_processes = os.cpu_count()
    file_lists = [file_list[i::num_processes] for i in range(num_processes)]

    # Create a pool for processing
    with Pool(processes=num_processes) as pool:
        pool.map(convert_xml2abc, file_lists)