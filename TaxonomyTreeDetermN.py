import spacy
from nltk.corpus import wordnet as wn
import time

nlp = spacy.load("en_core_web_sm")

start_time = time.time()

class Node:
    def __init__(self, name):
        self.name = name
        self.children = []

# Function to check if a word is a single word
def is_single_word(word):
    return "_" not in word

# Function to get related single words for a given word
def get_related_words(word):
    related_words = set()
    for syn in wn.synsets(word):
        for hypernym in syn.hypernyms():
            for lemma in hypernym.lemmas():
                related_word = lemma.name()
                if is_single_word(related_word):
                    related_words.add(related_word)
        for hyponym in syn.hyponyms():
            for lemma in hyponym.lemmas():
                related_word = lemma.name()
                if is_single_word(related_word):
                    related_words.add(related_word)
        for lemma in syn.lemmas():
            synonym = lemma.name()
            if is_single_word(synonym):
                related_words.add(synonym)
    return sorted(list(related_words))  # Sort the related words alphabetically

# Function to add children nodes with related words while ensuring uniqueness
def add_children_nodes(node, max_children_per_node, visited):
    related_words = get_related_words(node.name)
    children_count = 0
    for word in related_words:
        if children_count >= max_children_per_node:
            break
        if word not in visited:
            child_node = Node(word)
            node.children.append(child_node)
            visited.add(word)
            children_count += 1

# Function to create the taxonomy tree using breadth-first search
def create_taxonomy_tree(root_word, max_children_per_node, max_depth):
    root = Node(root_word)
    visited = set([root_word])
    queue = [(root, 1)]

    while queue:
        current_node, depth = queue.pop(0)
        if depth < max_depth:
            add_children_nodes(current_node, max_children_per_node, visited)
            for child in current_node.children:
                queue.append((child, depth + 1))

    return root

# Function to create a text-based representation of the taxonomy tree with level numbers
def print_nodes(node, level=1):
    graph_representation = f"{level} : {node.name}\n"
    for child in node.children:
        graph_representation += print_nodes(child, level + 1)
    return graph_representation

# Ask user for the root word, max children per node, and max depth
root_word = input("Enter the root word for the taxonomy tree: ")
max_children_per_node = int(input("Enter the maximum number of children nodes per node: "))
max_depth = int(input("Enter the maximum depth for the taxonomy tree: "))

# Create the taxonomy tree based on user input with specified max_children_per_node and max_depth
root_node = create_taxonomy_tree(root_word, max_children_per_node, max_depth)

# Generate the text-based representation of the taxonomy tree with level numbers
graph_text_representation = print_nodes(root_node)

# Write the graph text representation to a file
output_file = "taxonomy_tree_updated.txt"
with open(output_file, "w") as file:
    file.write(graph_text_representation)

print(f"Updated text-based graph representation of the taxonomy tree with level numbers has been written to {output_file}")

# Display runtime
print(f"\nScript execution time: {time.time() - start_time} seconds")