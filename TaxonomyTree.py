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
    return list(related_words)

# Function to recursively add children nodes
def add_children_nodes(node, level, max_levels, visited):
    if level >= max_levels:
        return

    related_words = get_related_words(node.name)
    for word in related_words:
        if word not in visited:
            child_node = Node(word)
            node.children.append(child_node)
            visited.add(word)
            add_children_nodes(child_node, level + 1, max_levels, visited)

# Function to create the taxonomy tree
def create_taxonomy_tree():
    root_word = input("Enter the root word for the taxonomy tree: ")
    root = Node(root_word)
    visited = set([root_word])
    
    add_children_nodes(root, 1, 5, visited)
    
    return root

# Function to write the tree to a file
def write_tree_to_file(node, filename):
    with open(filename, 'w') as file:
        write_tree_recursive(node, file)

# Function to recursively write the tree
def write_tree_recursive(node, file, level=0):
    if node:
        file.write("  " * level + node.name + "\n")
        for child in node.children:
            write_tree_recursive(child, file, level + 1)

# Usage
root_node = create_taxonomy_tree()
write_tree_to_file(root_node, "taxonomy_tree.txt")

# Display runtime
print(f"Script execution time: {time.time() - start_time} seconds")