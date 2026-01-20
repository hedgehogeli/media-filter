import json
import sys
import os
import pyperclip  # install with: pip install pyperclip

def ipynb_to_py(ipynb_path, output_path=None):
    # Load the notebook
    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Extract code cells
    code_cells = []
    for i, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") == "code":
            cell_content = "".join(cell.get("source", []))
            
            # Check if cell contains "# ignore below"
            if "# ignore below" in cell_content.strip():
                print(f"Stopped at cell {i}: found '# ignore below'")
                break
            
            code_cells.append(cell_content)

    # Prepare final script content
    header = f"# {os.path.basename(ipynb_path)}\n\n"
    script_content = header + "\n\n".join(code_cells)

    if output_path:
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        print(f"Extracted {len(code_cells)} code cells into {output_path}")
    else:
        # Copy to clipboard
        pyperclip.copy(script_content)
        print(f"Extracted {len(code_cells)} code cells into clipboard")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ipynb_to_py.py <input_notebook.ipynb> [output_file.py]")
    elif len(sys.argv) == 2:
        ipynb_to_py(sys.argv[1])  # clipboard mode
    else:
        ipynb_to_py(sys.argv[1], sys.argv[2])  # file mode