import json
import os
import string
import numpy as np
import argparse

def parse_suguru_json_file(filename):
    with open(filename, 'r') as fp:
        content = json.load(fp)
        for puzzle in content:
            id = puzzle['id']
            rows, cols = puzzle["rows"], puzzle["cols"]

            # Create empty grid
            grid = np.zeros((rows, cols), dtype=np.int8)
            regions = np.zeros((rows, cols), dtype=np.int8)

            # Fill grid
            cells_str = puzzle["cells"]
            for idx in range(rows * cols):
                val = int(cells_str[2*idx])
                row, col = divmod(idx, cols)
                grid[row, col] = val

            # Fill regions
            for region_id, group_str in enumerate(puzzle["groups"]):
                for i in range(0, len(group_str), 2):
                    cell_idx = int(group_str[i:i+2])
                    r, c = divmod(cell_idx, cols)
                    regions[r, c] = region_id

            yield id, grid, regions


def to_binary_file(output_path, id, grid, regions, solution=None):
    rows, cols = grid.shape
    if solution is None:
        conc = np.vstack([grid, grid, regions])
    else:
        conc = np.vstack([grid, solution, regions])
    path = os.path.join(output_path, f'{rows}x{cols}_{id}.data')

    with open(path, 'wb') as fp:
        fp.write(rows.to_bytes(2))
        fp.write(cols.to_bytes(2))
        conc.astype('int16').tofile(fp)


def parse_suguru_txt_file(file_path):
    with open(file_path, 'r') as fp:
        content = fp.readlines()
        id = 1
        for line in content[1:]:
            name, w, h, clues, layout, answer, comments = line.strip().split("\t")
            name = f'{name}_{id}'
            id += 1
            w, h = int(w), int(h)

            # --- Regions (map A,B,C... -> 1,2,3...) ---
            letters = sorted(set(layout), key=lambda x: layout.index(x))
            letter_to_id = {ch: i + 1 for i, ch in enumerate(letters)}
            regions = np.array([letter_to_id[ch] for ch in layout]).reshape(h, w)

            # --- Clues / givens ---
            letter_to_skip = {ch: i + 1 for i, ch in enumerate(string.ascii_lowercase)}

            givens = np.zeros((h, w), dtype=int)
            idx = 0
            i = 0
            while i < len(clues):
                ch = clues[i]
                if ch.isdigit():
                    r, c = divmod(idx, w)
                    givens[r, c] = int(ch)
                    idx += 1
                    i += 1
                else:
                    idx += letter_to_skip[ch]
                    i += 1

            # --- Solution ---
            solution = np.array([int(x) for x in answer]).reshape(h, w)

            yield name, givens, solution, regions



def main():
    parser = argparse.ArgumentParser(description='Parse Suguru JSON file to binary')
    parser.add_argument('--path', help='Suguru JSON or TXT file to parse')
    parser.add_argument('--output', help='Output directory for extracted puzzles')
    args = parser.parse_args()

    file_path = args.path
    output_path = args.output

    if file_path.endswith('.json'):
        for id, grid, regions in parse_suguru_json_file(file_path):
            to_binary_file(output_path, id, grid, regions)
    elif file_path.endswith('.txt'):
        for id, grid, solution, regions in parse_suguru_txt_file(file_path):
            to_binary_file(output_path, id, grid, regions, solution)

if __name__ == '__main__':
    main()