#!/usr/bin/env python
# coding: utf-8

# ### Mesh agent

import os
import openai
from dotenv import load_dotenv, find_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import oracledb
# from difflib import SequenceMatcher
# import Levenshtein as lev
# from fuzzywuzzy import process
from transformers import AutoModel
import numpy as np

_ = load_dotenv(find_dotenv()) #read local.env file
openai.api_key = os.environ['OPENAI_API_KEY']

class FailureModeAgent:
    """
    A class representing an agent for suggesting failure modes based on a given description.

    Attributes:
        model: The model used for generating failure mode suggestions.
        connection: The connection to the Oracle database.

    Methods:
        suggest_failure_modes: Generates failure mode suggestions based on a given description by the technician of his observations while executing the work order.
        preprocess: Preprocesses the description before generating suggestions.
        postprocess: Postprocesses the generated suggestions.
        get_failure_modes: Retrieves the failure modes from the Oracle database.
    """

    def __init__(self, model):
        """
        Initializes the FailureModeAgent with a given model and database connection.
        """
        self.model = model
        self.connection = oracledb.connect(user="oracle", password="trlmTRLM14", dsn="172.21.207.73:1539/xepdb1")
        
    def format_match(self, match):
        id, part, damage, cause, consequence, *effects = match
        id = str(id).replace(" ", "").replace("'", "")
        effects = ' '.join(effects)
        return f'{id} "{part}" "{damage}" "{cause}" "{consequence}" {effects}'
    
    def suggest_failure_modes(self, description, equipment):
        # print(f"Description: {description}")
        print(f"Equipment entered: {equipment}")

        # Convert the description to lower case
        description = description.lower()

        # Preprocess the description
        # description = self.preprocess(description)
        # print(f"Preprocessed description: {description}")
        
        # Get the closest equipment name from the database
        closest_equipment_name = self.get_closest_equipment_name(equipment)
        print(f"Closest equipment name: {closest_equipment_name}")

        # Get the failure modes for the closest equipment
        failure_modes = self.get_failure_modes(closest_equipment_name)
        # print(f"Failure modes: {failure_modes}")

        # Convert the failure modes to lower case
        failure_modes = [fm.lower() for fm in failure_modes]

        # Combine the description with the failure modes
        # documents = [description] + [mode[1] for mode in failure_modes]
        
        # Replace linefeeds with spaces in the failure modes
        failure_modes = [mode.replace('\n', ' ') for mode in failure_modes]

        # Replace linefeeds with spaces in the description
        description = description.replace('\n', ' ')
        
        # Tokenize the documents
        tokenized_failure_modes = [doc.split(" ") for doc in failure_modes]

        # Check if tokenized_failure_modes is not empty (to avoid div / 0)
        if tokenized_failure_modes:
            # Create a BM25 object
            bm25 = BM25Okapi(tokenized_failure_modes)
        

        # Compute the BM25 scores for the description against the failure modes
        scores = bm25.get_scores(description.split(" "))

        # Get the indices of the scores sorted in descending order
        sorted_indices = np.argsort(scores)[::-1]

        # Get the top N matches
        top_n = 5
        top_matches = [(failure_modes[i], scores[i]) for i in sorted_indices[:top_n]]

        # Convert each tuple to a string and join them with newline characters
        top_matches_str = '\n'.join(str(match) for match in top_matches)

        return top_matches_str

    # Define a function to format each match
    def format_match(match):
        # If the match is a tuple, convert each item to a string and join them with a space
        if isinstance(match[0], tuple):
            match_string = ' '.join(str(item).strip() for item in match[0])
        else:  # If the match is not a tuple, simply convert it to a string
            match_string = str(match[0]).strip()
        # Append the score to the match string
        match_string += " (Score: " + str(match[1]) + ")"
        return match_string

        # Print the top matches
        if top_matches:
            matches_string = "\n".join(format_match(match) for match in top_matches)
            result = "\n\033[93mBased on your observation I found the following top " + str(top_n) + " matches:\033[0m\n" + matches_string
            print(result)
        else:
            return "No match found. You may submit a new failure mode record to add to RCM knowledge base."

    def get_closest_equipment_name(self, equipment):
        """
        Finds the closest match for a given equipment name in the database.

        Args:
            equipment: The equipment name provided by the technician.

        Returns:
            The closest match for the equipment name in the database.
        """
        
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM mesh_ug.fleets ORDER BY name")
        
        query_result = cursor.fetchall()
        # print("SQL query result:", query_result)
        
        original_equipment_names = [row[0] for row in query_result]  # Store the original equipment names
        equipment_names = [name.lower() for name in original_equipment_names]  # Convert to lowercase
        # print("Equipment names:", equipment_names)
        
        # Tokenize the equipment names
        tokenized_equipment_names = [name.split(" ") for name in equipment_names]

        # Create a BM25 object
        bm25 = BM25Okapi(tokenized_equipment_names)

        # Compute the BM25 scores for the equipment against the equipment names
        scores = bm25.get_scores(equipment.lower().split(" ")) # Convert to lowercase

        # Find the index of the most similar equipment name
        most_similar_index = np.argmax(scores)

        if scores[most_similar_index] >= 0.5:  # Allow down to 50% similarity
            # print(f"Found equipment: {original_equipment_names[most_similar_index]}")
            return original_equipment_names[most_similar_index] # Return the original case equipment name
        else:
            print("Equipment not found.")
            return equipment
        
    def preprocess(self, description):
        """
        Preprocesses the description before generating suggestions.

        Args:
            description: The description of the failure.

        Returns:
            The preprocessed description.
        """
        # Implement preprocessing steps here
        return description

    def postprocess(self, prediction):
        """
        Postprocesses the generated suggestions.

        Args:
            prediction: The generated prediction.

        Returns:
            The postprocessed prediction.
        """
        # Implement postprocessing steps here
        return prediction

    def get_failure_modes(self, equipment):
        """
        Retrieves the failure mode full RCM records from the Oracle database.

        Args:
            equipment: The equipment that the technician has been working on.

        Returns:
            A list of failure modes and defined by their respective ID, part, damage, cause, consequence, and effects.
        """
        cursor = self.connection.cursor()
        cursor.execute(f"""
            SELECT fm.id, ae_part.name AS part, ae_damage.name AS damage, ae_cause.name AS cause,  fc.name AS consequence, REPLACE(REPLACE(fm.effects, CHR(10), ' '), CHR(13), ' ')
            FROM mesh_ug.failure_modes fm
            JOIN mesh_ug.knowledge_nodes kn ON fm.id = kn.failure_mode_id
            JOIN mesh_ug.knowledge_trees kt ON kn.tree_id = kt.id
            JOIN mesh_ug.fleets f ON kt.fleet_id = f.id
            JOIN mesh_ug.analisys_elements ae_cause ON fm.cause_id = ae_cause.id
            JOIN mesh_ug.analisys_elements ae_damage ON fm.damage_id = ae_damage.id
            JOIN mesh_ug.failure_consequences fc ON fm.consequence_id = fc.id
            JOIN mesh_ug.analisys_elements ae_part ON fm.part_id = ae_part.id AND ae_part.type = 'FAILURE_PART'
            WHERE f.name = '{equipment}'
        """)
        rows = cursor.fetchall()
        
        # print("Results of query\n")
        # for row in rows:
        #    print(row)

        # Convert each row from the query result into a string by joining all the elements in the row with a space (' ')
        failure_modes = [' '.join(str(item) for item in row) for row in rows]

        return failure_modes

class AgentExecutor:
    def __init__(self, agent):
        self.agent = agent

    def get_user_input(self, technician_name=None):
        if not technician_name:
            technician_name = input("\033[93m" + "Please enter your name: " + "\033[0m")
        equipment = input(f"\033[93m\nHow can I help you, {technician_name}? Can you begin by telling me which equipment you are interested in? \n\033[0m")
        print(f"{technician_name} is working on: {equipment}")
        description = input("\033[93m" + "\nAI Agent: Would you describe your observations and other details about this job? \n" + "\033[0m")
        return  technician_name, equipment, description

    def execute_agent(self, equipment, description):
        try:
            suggestions = self.agent.suggest_failure_modes(description, equipment)
            print("\033[93m" + "\nSuggested failure modes:\n" + "\033[0m", suggestions)
            return True
        except Exception as e:
            print(f"\033[91mInvalid input: {e}. Please try again with a closer description to your equipment or observation.\033[0m")
            import traceback
            traceback.print_exc() # print the full traceback
            return False

# A pre-trained BERT model and its corresponding tokenizer are loaded from the transformers library, and these are passed to the FailureModeAgent during initialization.

def main():
    # Load the pre-trained BERT model
    model_name = 'bert-base-uncased'
    model = AutoModel.from_pretrained(model_name)

    # Initialize the agent and executor
    agent = FailureModeAgent(model)
    executor = AgentExecutor(agent)

    technician_name = None
    while True:
        # Get the user's input
        technician_name, equipment, description = executor.get_user_input(technician_name)

        # Execute the agent's suggestion method
        if executor.execute_agent(equipment, description):
            while True:
                # Ask the user for their choice
                print("\033[93m" + "Please select an option:" + "\033[0m")
                print("1. Increment the history with this failure mode instance")
                print("2. Modify the RCM record")
                print("3. Add a new RCM record")
                print("4. Continue")
                print("5. Quit")
                user_choice = input("\033[93m" + "Your choice: " + "\033[0m")

                if user_choice in ["1", "2", "3"]:
                    print("Not implemented yet. Please select 4 or 5.")
                elif user_choice == "4":
                    break
                elif user_choice == "5":
                    return
                else:
                    print("Invalid choice. Please enter a number between 1 and 5.")

main()

