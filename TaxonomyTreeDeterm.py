import spacy
from nltk.corpus import wordnet as wn
import time
import networkx as nx
import matplotlib.pyplot as plt

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

# Function to recursively add children nodes in a deterministic order
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
def create_taxonomy_tree(root_word):
    root = Node(root_word)
    visited = set([root_word])

    add_children_nodes(root, 1, 5, visited)

    return root

# Function to create a graph representation of the taxonomy tree
def create_graph(node, G=None):
    if G is None:
        G = nx.DiGraph()
    G.add_node(node.name)
    for child in node.children:
        G.add_edge(node.name, child.name)
        create_graph(child, G)
    return G

# Ask user for the root word
root_word = input("Enter the root word for the taxonomy tree: ")

# Create the taxonomy tree based on user input
root_node = create_taxonomy_tree(root_word)

# Create a graph representation of the taxonomy tree
G = create_graph(root_node)

# Visualize the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray", arrowsize=20)
plt.title("Taxonomy Tree Graph Visualization")
plt.axis("off")
plt.show()

# Display runtime
print(f"\nScript execution time: {time.time() - start_time} seconds")