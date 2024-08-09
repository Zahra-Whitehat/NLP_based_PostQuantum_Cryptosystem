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
        print(f"\nScript execution time: {time.time() - self.start_time} seconds")
    '''  with open(output_file, "a") as file:
            file.write("\n\nIDs of the input words in the taxonomy tree:\n")
            for word in input_words:
                node = self.get_node_by_identifier(word)
                if node:
                    file.write(f"{word}: {node.identifier}\n")
                else:
                    file.write(f"{word}: No corresponding node found\n")'''
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
    def dbscan_clustering(self, eps, min_samples):
        nodes = list(self.node_identifier_map.values())
        
        if not nodes:
            print("No nodes available for clustering. Ensure there are nodes in the node_identifier_map.")
            return
        
        # Convert nodes to a 2D array
        nodes_array = np.array(nodes).reshape(-1, 1)
        
        # Perform DBSCAN clustering on the reshaped array
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(nodes_array)

        # Assign cluster labels to nodes
        for i, node in enumerate(nodes):
            node.cluster_label = cluster_labels[i]


taxonomy_tree = TaxonomyTree()
taxonomy_tree.dbscan_clustering(eps=0.5, min_samples=5)  # Example parameters for DBSCAN
taxonomy_tree.run()
