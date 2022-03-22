import triton
import pickle

class Node:
    def __init__(self, lineno, col_offset):
        self.lineno = lineno
        self.col_offset = col_offset

src = ""
node = Node(0, 0)
error = triton.code_gen.CompilationError(src, node)

pickled = pickle.dumps(error)
original = pickle.loads(pickled)
