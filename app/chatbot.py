import argparse
from pathlib import Path
from chatbot import get_answer
from view import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Chatbot',
                    description='Run a chatbot using a semantic search system',
                    epilog='Jose Angel Perez Garrido - 2023')
    parser.add_argument("-o", "--option", type=str, help="select an option: retrieve_topk_answer -> get topk answers from given configuration; cli -> run a chatbot in command line view. (default: cli)", required=True)
    parser.add_argument("-m", "--model", type=str, help="select a model", default='publichealthsurveillance/PHS-BERT')
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalFAQ")
    parser.add_argument("-f", "--fields", type=str, help="select rank fields separated by commas")
    parser.add_argument("-q", "--question", type=str, help="introduce a mental health related question")
    args = parser.parse_args()

    path = Path.cwd()

    if args.option == "retrieve_topk_answer":
        # python app\chatbot.py -o retrieve_topk_answer -q "What is a mental illness?" -d MentalFAQ -m publichealthsurveillance/PHS-BERT -f question
        top_k = 3
        dataset = args.dataset
        model = args.model
        fields = [str(item) for item in args.fields.split(',')]
        model_path = dataset+"/models/"+model

        # Elasticsearch index name
        index_name = dataset.lower()

        # Define model parameters
        loss_type = 'softmax'; neg_type = 'simple'; query_type = 'faq'

        # Question
        question = args.question

        # Get model path
        version = '1.1'
        model_name = "{}_{}_{}_{}".format(loss_type, neg_type, query_type, version)
        bert_model_path = "output" + "/" + model_path + "/models/" + model_name

        answer = get_answer(question)

        print(answer)

    if args.option == "cli":
        # python app\chatbot.py -o cli
        dataset = F"{str(path)}/file/data/MentalFAQ/query_answer_pairs.json"

        ask_sentence(dataset)

    print("PROGRAM FINISHED")