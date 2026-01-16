from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# New Python sample code
code = '''
# file_handler.py

def read_file(filepath):
    """Read the contents of a text file."""
    try:
        with open(filepath, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."

def write_file(filepath, content):
    """Write content to a text file."""
    with open(filepath, 'w') as file:
        file.write(content)

class FileManager:
    """Handles file operations."""

    def __init__(self, path):
        self.path = path

    def save(self, data):
        write_file(self.path, data)

    def load(self):
        return read_file(self.path)

if __name__ == "__main__":
    fm = FileManager("example.txt")
    fm.save("Hello, world!")
    print(fm.load())
'''

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=0
)

result = splitter.split_text(code)

print(result[2])
