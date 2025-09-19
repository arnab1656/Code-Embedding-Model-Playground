from Salesforce.sfr_embedding_code_400m_r_model import embedding_retreival


query = ["How to calculate hcf in python"]
passages = [
          # String operations
          "def reverse_string(s):\n    return s[::-1]",
          "def is_palindrome(s):\n    return s == s[::-1]",
          "def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')",

          # List operations
          "def find_max(nums):\n    return max(nums)",
          "def remove_duplicates(lst):\n    return list(set(lst))",
          "def flatten_list(lst):\n    return [item for sublist in lst for item in sublist]",

          # Math utilities
          "def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)",
          "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
          "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
          
          # File and misc
          "def read_file(filename):\n    with open(filename, 'r') as f:\n        return f.read()",
          "def write_file(filename, text):\n    with open(filename, 'w') as f:\n        f.write(text)",
          "def is_even(n):\n    return n % 2 == 0",
      ]

if __name__ ==  "__main__" :
  
  best_passage, top_k_retreival = embedding_retreival(query, passages)

print("best_passage --> \n",best_passage)

print("top_k_retreival --> \n",top_k_retreival)