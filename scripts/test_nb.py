import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_code_cell('x = 42\nprint("Result:", x)'))
client = NotebookClient(nb, timeout=60)
client.execute()
print("Execution success!")
print("Cell outputs:", nb.cells[0].outputs)
