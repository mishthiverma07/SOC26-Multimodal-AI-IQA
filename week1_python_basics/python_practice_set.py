# Python Practice Set
# Based on Week 1 Python basics resources:
# print statements, variables, strings, lists, dictionaries, sets, loops, functions, and OS module.

import os

# 1. Print a poem using triple quotes
print("""
Twinkle, twinkle, little star,
How I wonder what you are!
Up above the world so high,
Like a diamond in the sky.
""")

# 2. Print multiplication table
number = 5

print("Multiplication table of", number)
for i in range(1, 11):
    print(number, "x", i, "=", number * i)

# 3. Use OS module to list files in current directory
print("\nFiles and folders in current directory:")
items = os.listdir(".")
for item in items:
    print(item)

# 4. Variables and data types
name = "Python"
version = 3
is_programming_language = True

print("\nVariable examples:")
print("Name:", name)
print("Version:", version)
print("Is programming language:", is_programming_language)

# 5. String operations
sentence = "python is useful for machine learning"

print("\nString operations:")
print(sentence.upper())
print(sentence.title())
print(sentence.replace("machine learning", "AI"))

# 6. Lists
subjects = ["Python", "Pandas", "Machine Learning", "Deep Learning"]

print("\nList items:")
for subject in subjects:
    print(subject)

subjects.append("LLM")
print("Updated list:", subjects)

# 7. Dictionary
student_progress = {
    "Python": "basic practice done",
    "Machine Learning": "started",
    "Deep Learning": "not started yet"
}

print("\nDictionary items:")
for topic, status in student_progress.items():
    print(topic, ":", status)

# 8. Sets
set_a = {1, 2, 3, 4}
set_b = {3, 4, 5, 6}

print("\nSet operations:")
print("Union:", set_a.union(set_b))
print("Intersection:", set_a.intersection(set_b))

# 9. Function
def greet(topic):
    return "I am learning " + topic

print("\nFunction output:")
print(greet("Python basics"))