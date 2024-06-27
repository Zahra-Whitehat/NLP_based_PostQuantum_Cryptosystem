import spacy
from nltk.corpus import wordnet as wn
import time

nlp = spacy.load("en_core_web_sm")

class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.positions = []

def is_single_word(word):
    return "_" not in word

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
    return sorted(list(related_words))

def add_children_nodes(node, max_children_per_node, visited, pos):
    related_words = get_related_words(node.name)
    children_count = 0
    for word in related_words:
        if children_count >= max_children_per_node:
            break
        if word not in visited:
            child_node = Node(word)
            child_node.positions = node.positions + [pos]
            node.children.append(child_node)
            visited.add(word)
            children_count += 1

def create_taxonomy_tree(root_word, max_children_per_node, max_depth):
    root = Node(root_word)
    root.positions = [0]
    visited = set([root_word])
    queue = [(root, 1)]

    while queue:
        current_node, depth = queue.pop(0)
        if depth < max_depth:
            add_children_nodes(current_node, max_children_per_node, visited, depth)
            for child in current_node.children:
                queue.append((child, depth + 1))

    return root

def print_nodes(node, level=1):
    graph_representation = f"{level} : {node.name} - Positions: {node.positions}\n"
    for child in node.children:
        graph_representation += print_nodes(child, level + 1)
    return graph_representation

def extract_topic_from_phrase(phrase):
    doc = nlp(phrase)
    for token in doc:
        if token.pos_ == "NOUN":
            return token.text
    return None

start_time = time.time()

phrase = input("Enter the phrase to extract the topic from: ")
topic_word = extract_topic_from_phrase(phrase)
if topic_word is None:
    print("Unable to extract a topic from the phrase.")
    exit()

max_children_per_node = 3
max_depth = 3

root_node = create_taxonomy_tree(topic_word, max_children_per_node, max_depth)
graph_text_representation = print_nodes(root_node)

output_file = "taxonomy_tree_updated.txt"
with open(output_file, "w") as file:
    file.write(graph_text_representation)

print(f"Updated text-based graph representation of the taxonomy tree with word positions has been written to {output_file}")
print(f"Positions of the original phrase words in the tree: {root_node.positions}")

print(f"\nScript execution time: {time.time() - start_time} seconds")