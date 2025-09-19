from Salesforce.sfr_embedding_code_400m_r_model import embedding_retreival

from langchain_text_splitters import RecursiveCharacterTextSplitter

PYTHON_CODE = [
"""
class GridSolver:
    def min_path_sum(self, grid):
        if not grid or not grid[0]:
            return 0
        m, n = len(grid), len(grid[0])
        dp = [[0]*n for _ in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]

class MathUtils:
    def factorial(self, n):
        return 1 if n == 0 else n * self.factorial(n-1)

class StringUtils:
    def reverse_string(self, s: str) -> str:
        return s[::-1]

    def is_palindrome(self, s: str) -> bool:
        return s == s[::-1]

class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b

    def multiply(self, a: int, b: int) -> int:
        return a * b

    def divide(self, a: int, b: int) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

class FileHandler:
    def __init__(self, filename):
        self.filename = filename

    def read_file(self):
        with open(self.filename, "r") as f:
            return f.read()

    def write_file(self, content):
        with open(self.filename, "w") as f:
            f.write(content)

class TemperatureConverter:
    def celsius_to_fahrenheit(self, c: float) -> float:
        return (c * 9/5) + 32

    def fahrenheit_to_celsius(self, f: float) -> float:
        return (f - 32) * 5/9
"""
]


# custom_separators = ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']

# custom_separators = ['\nclass ', '\ndef ', '\n\tdef ', '\n\n']

custom_separators = ['\nclass ', '\ndef ']

python_splitter = RecursiveCharacterTextSplitter(
    separators=custom_separators,
    keep_separator=True,
    is_separator_regex=False,
    chunk_size=50,
    chunk_overlap=0
)

python_docs = python_splitter.create_documents(PYTHON_CODE)

for i, docs in enumerate(python_docs):
  print("At index ---> ",i+1)
  print("docs is ----> \n\n",docs.page_content,"\n\n")

python_docs_text = [p.page_content for p in python_docs]

query = "calculate multiply two number in python"

if __name__ ==  "__main__" :
  
  best_passage, top_k_retreival = embedding_retreival(query, python_docs_text)

print("best_passage --> \n" ,best_passage)


print("top_k_retreival --> \n" ,top_k_retreival)