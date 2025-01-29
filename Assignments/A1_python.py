# 1. String Length
s = "Hello, World!"
print(sum(1 for _ in s))  # Using generator expression

# 2. Convert to Uppercase
print("".join([ch.upper() for ch in s]))  # Using list comprehension

# 3. Convert to Lowercase
print("".join(map(str.lower, s)))  # Using map function

# 4. Capitalize First Letter
print(s[:1].upper() + s[1:].lower())  # Using slicing

# 5. Title Case
print(" ".join([word.capitalize() for word in s.split()]))  # Using split & capitalize

# 6. Swap Case
print("".join(ch.lower() if ch.isupper() else ch.upper() for ch in s))  # Using generator expression

# 7. Count Occurrences of a Substring
print(len([c for c in s if c == "o"]))  # Using list comprehension

# 8. Find Index of Substring
print(next((i for i in range(len(s)) if s[i:i+5] == "World"), -1))  # Using generator

# 9. Replace a Substring
print(s[:7] + "Python" + s[12:])  # Using slicing

# 10. Check if String Starts with a Specific Substring
print(s[:5] == "Hello")  # Using slicing

# 11. Check if String Ends with a Specific Substring
print(s[-1] == "!")  # Using indexing

# 12. Split String into List
print([word for word in s.split(",")])  # Using list comprehension

# 13. Join List into String
from functools import reduce
words = ["Python", "is", "awesome"]
print(reduce(lambda x, y: x + " " + y, words))  # Using reduce

# 14. Remove Leading and Trailing Spaces
s2 = "   Trim Spaces   "
print(" ".join(s2.split()))  # Using split & join

# 15. Check if String Contains Only Digits
num_str = "12345"
print(all(ch in "0123456789" for ch in num_str))  # Using all()

# 16. Check if String Contains Only Alphabets
alpha_str = "Python"
print(all(c.isalpha() for c in alpha_str))  # Using all()

# 17. Check if String Contains Alphanumeric Characters
alnum_str = "Python123"
print(any(c.isdigit() for c in alnum_str) and any(c.isalpha() for c in alnum_str))  # Using any()

# 18. Check if String is in Lowercase
print(all(not c.isupper() for c in s if c.isalpha()))  # Using all()

# 19. Check if String is in Uppercase
print(all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in s if c.isalpha()))  # Using set

# 20. Check if String is Title Case
print(s == " ".join([w.capitalize() for w in s.split()]))  # Using list comprehension

# 21. Check if String Contains Only Whitespace
space_str = "   "
print(space_str.strip() == "")  # Using strip()

# 22. Center Align String
print(s.rjust(19, "-").ljust(20, "-"))  # Using rjust & ljust

# 23. Left Align String
print(s + "-" * (20 - len(s)))  # Using concatenation

# 24. Right Align String
print("-" * (20 - len(s)) + s)  # Using concatenation

# 25. Convert Tabs to Spaces
tab_str = "Python\tis\tfun"
print(tab_str.replace("\t", "    "))  # Using replace()
