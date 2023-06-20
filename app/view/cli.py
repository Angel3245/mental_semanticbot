from InquirerPy import inquirer
from shared import load_from_json
from chatbot import get_answer

# import random 
from random import sample

# Disable log messages
import logging
logging.disable(logging.ERROR)

def ask_sentence(dataset):
    # Ask user to select between a predefined post or a custom question 
    mode = inquirer.select(
        message="Select mode:",
        choices=["Introduce a question", "Choose a recommended question"],
    ).execute()

    # Get question
    if(mode == "Introduce a question"):
        # Keyboard input
        question = inquirer.text(message="Introduce sentence:").execute()
    elif(mode == "Choose a recommended question"):
        questions = load_from_json(dataset)

        # choose one from 3 random questions
        question = inquirer.select(
            message="Recommended questions:",
            choices=sample([question['question'] for question in questions],3),
        ).execute()
    else:
        raise ValueError("Selected mode does not exist")
    
    answer = get_answer(question)
    
    print(answer)
    