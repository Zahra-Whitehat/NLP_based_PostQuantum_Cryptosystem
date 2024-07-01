import spacy
from nltk.corpus import wordnet as wn
import time

class TaxonomyTree:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.start_time = time.time()
        self.node_identifier_map = {}

    class Node:
        def __init__(self, name, identifier):
            self.name = name
            self.children = []
            self.parent = None
            self.identifier = identifier

    def is_single_word(self, word):
        return "_" not in word

    def get_related_words(self, word):
        related_words = set()
        for syn in wn.synsets(word):
            for hypernym in syn.hypernyms():
                for lemma in hypernym.lemmas():
                    related_word = lemma.name()
                    if self.is_single_word(related_word):
                        related_words.add(related_word)
            for hyponym in syn.hyponyms():
                for lemma in hyponym.lemmas():
                    related_word = lemma.name()
                    if self.is_single_word(related_word):
                        related_words.add(related_word)
            for lemma in syn.lemmas():
                synonym = lemma.name()
                if self.is_single_word(synonym):
                    related_words.add(synonym)
        return sorted(list(related_words))

    def add_children_nodes(self, node, max_children_per_node, visited):
        related_words = self.get_related_words(node.name)
        children_count = 0
        for word in related_words:
            if children_count >= max_children_per_node:
                break
            if word not in visited:
                child_node = self.Node(word, len(self.node_identifier_map))  # Assign unique identifier
                self.node_identifier_map[len(self.node_identifier_map)] = child_node
                child_node.parent = node
                node.children.append(child_node)
                visited.add(word)
                children_count += 1

    def create_taxonomy_tree(self, root_word, max_children_per_node, max_depth):
        root = self.Node(root_word, len(self.node_identifier_map))  # Assign unique identifier to root node
        self.node_identifier_map[len(self.node_identifier_map)] = root
        visited = set([root_word])
        queue = [(root, 1)]
        while queue:
            current_node, depth = queue.pop(0)
            if depth < max_depth:
                self.add_children_nodes(current_node, max_children_per_node, visited)
                for child in current_node.children:
                    queue.append((child, depth + 1))
        return root

    def get_node_by_identifier(self, identifier):
        return self.node_identifier_map.get(identifier)

    def run(self):
        input_phrase, main_topic = self.input_phrase()
        root_word = input("Enter the root word for the taxonomy tree: ")
        max_children_per_node = int(input("Enter the maximum number of children nodes per node: "))
        max_depth = int(input("Enter the maximum depth for the taxonomy tree: "))
        root_node = self.create_taxonomy_tree(root_word, max_children_per_node, max_depth)
        graph_text_representation = self.print_nodes(root_node)
        output_file = "taxonomy_tree_updated.txt"
        with open(output_file, "w") as file:
            file.write(graph_text_representation)
        print(f"Updated text-based graph representation of the taxonomy tree with parent and child nodes has been written to {output_file}")
        if main_topic:
            print(f"\nMain topic extracted from the input phrase: {main_topic}")
        else:
            print("\nMain topic could not be extracted from the input phrase.")
        print(f"\nScript execution time: {time.time() - self.start_time} seconds")

    def print_nodes(self, node, parent="", level=1):
        graph_representation = f"({level},{parent}) : {node.name} (ID: {node.identifier})\n"
        for child in node.children:
            graph_representation += self.print_nodes(child, node.name, level + 1)
        return graph_representation

    def input_phrase(self):
        phrase = input("Enter a phrase: ")
        doc = self.nlp(phrase)
        main_topic = None
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON", "GPE"]:
                main_topic = ent.text
                break
        return phrase, main_topic

# Instantiate and run the TaxonomyTree class
taxonomy_tree = TaxonomyTree()
taxonomy_tree.run()