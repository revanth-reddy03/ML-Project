import os
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def create_nb():
    return nbf.v4.new_notebook()

def add_md(nb, text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def add_code(nb, code):
    nb.cells.append(nbf.v4.new_code_cell(code))

def execute_and_save(nb, filepath):
    print(f"Executing and saving {filepath.name}...")
    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()
    with open(filepath, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated and executed {filepath.name}")

print("Helper functions defined.")
