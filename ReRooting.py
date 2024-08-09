import spacy
from nltk.corpus import wordnet as wn
import time
from sklearn.cluster import DBSCAN
import numpy as np

class TaxonomyTree:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.start_time = time.time()
        self.node_identifier_map = {}
        self.result = ""  # Initialize the result attribute

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
        self.root = root  # Set the root attribute here
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
        if not hasattr(self, 'root') or self.root is None:
            return None
        
        queue = [self.root]
        while queue:
            current_node = queue.pop(0)
            if current_node.identifier == identifier:
                return current_node
            queue.extend(current_node.children)
        return None

    def run(self):
        input_phrase, main_topic = self.input_phrase()
        root_word = input("Enter the root word for the taxonomy tree: ")
        max_children_per_node = int(input("Enter the maximum number of children nodes per node: "))
        max_depth = int(input("Enter the maximum depth for the taxonomy tree: "))
        root_node = self.create_taxonomy_tree(root_word, max_children_per_node, max_depth)
        
        input_words = input_phrase.split()
        
        # New code to scan the taxonomy tree and return IDs for input words
        word_ids_first_occurrence = self.scan_and_return_ids(input_words)
        
        graph_text_representation = self.print_nodes(root_node)
        output_file = "taxonomy_tree_updated.txt"
        with open(output_file, "w") as file:
            file.write("Text-based graph representation of the taxonomy tree with parent and child nodes:\n")
            file.write(graph_text_representation)
            file.write("\n\nWords and their corresponding IDs in the taxonomy tree:\n")
            for word, identifier in word_ids_first_occurrence.items():
                file.write(f"{word}: {identifier}\n")
        
        print(f"Updated text-based graph representation of the taxonomy tree and IDs of the words in the input phrase have been written to {output_file}")
        if main_topic:
            print(f"\nMain topic extracted from the input phrase: {main_topic}")
        else:
            print("\nMain topic could not be extracted from the input phrase.")
        # New code to add the function result to the end of the existing output file
        with open(output_file, "a") as file:
            new_root_id = int(input("Enter the ID of the new root node: "))
            self.change_root_and_relabel(new_root_id, output_file)
       
        print(f"\nScript execution time: {time.time() - self.start_time} seconds")
        output_file = "taxonomy_tree_updated.txt"

        # Write the result at the end of the file
        with open(output_file, "a") as file:
            self.result = "Your desired result here"
            file.write("\nResult:\n")
            file.write(self.result)  # Replace `self.result` with the actual result you want to write

        print(f"\nResult written to {output_file}")
        
        taxonomy_tree.change_root_and_relabel(new_root_id, "taxonomy_tree_updated.txt")

    def scan_and_return_ids(self, input_words):
        ids_found = {}
        queue = [self.root]
        
        while queue:
            current_node = queue.pop(0)
            for word in input_words:
                if current_node.name == word and word not in ids_found:
                    ids_found[word] = current_node.identifier
            
            queue.extend(current_node.children)
        
        return ids_found
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

    def change_root_and_relabel(self, new_root_id,output_file):
        if new_root_id not in self.node_identifier_map:
            raise ValueError("Node with the specified ID not found.")
        
        new_root = self.get_node_by_identifier(new_root_id)
        
        # Relabel nodes
        self.relabel_nodes(new_root)

        # Reform parent-child relations
        self.reform_parent_child_relations(new_root)

        # Update root
        self.root = new_root
        
        # Write the updated tree to the file
        with open(output_file, "a") as file:
            file.write(self.print_nodes(self.root))

    def relabel_nodes(self, node, identifier=0):
        node.identifier = identifier
        for child in node.children:
            identifier = self.relabel_nodes(child, identifier + 1)
        return identifier

    def reform_parent_child_relations(self, node):
        node.parent = None
        for child in node.children:
            child.parent = node
            self.reform_parent_child_relations(child)
            
taxonomy_tree = TaxonomyTree()
taxonomy_tree.run()
