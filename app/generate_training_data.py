import argparse
from pathlib import Path
from shared import *

from DB.connect import elasticsearch_connect
from semantic_search.BERT_FAQ import get_relevance_label_df, Hard_Negatives_Generator, Training_Data_Generator

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Generate training data',
                    description='Generate data needed for SBERT training',
                    epilog='Jose Angel Perez Garrido - 2023')
    parser.add_argument("-o", "--option", type=str, help="select an option: generate_hard -> Generate hard negative samples; generate_gt -> Generate triplet dataset for BERT finetuning", required=True)
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalFAQ")

    args = parser.parse_args()

    path = Path.cwd()

    # Generate hard negative samples
    if args.option == "generate_hard":
        # python app\generate_training_data.py -o generate_hard -d MentalFAQ
        if not args.dataset == "":
            data_path = F"{str(path)}/file/data/"+args.dataset

            query_answer_pair_filepath = data_path+'/query_answer_pairs.json'
            relevance_label_df = get_relevance_label_df(query_answer_pair_filepath) 

            es = elasticsearch_connect()
            index_name = args.dataset.lower()

            hng = Hard_Negatives_Generator(
                es=es, index=index_name, query_by=['question_answer'], top_k=100, query_type='faq')
            hard_negatives_faq = hng.get_hard_negatives(relevance_label_df)

            dump_to_json(hard_negatives_faq, data_path+'/hard_negatives_faq.json')
            print("Hard negative samples created")

        else:
            raise ValueError("Dataset not selected")

    # Generate triplet dataset for BERT finetuning
    if args.option == "generate_gt":
        # python app\generate_training_data.py -o generate_gt -d MentalFAQ
        if not args.dataset == "":
            data_path = F"{str(path)}/file/data/"+args.dataset

            # read query_answer_pairs
            filepath = data_path+'/query_answer_pairs.json'
            query_answer_pairs = load_from_json(filepath)

            # define variables
            for query_type in ["faq"]:
                for neg_type in ["simple"]: 
                    for loss_type in ["softmax"]: 
                        tdg = Training_Data_Generator(
                            random_seed=5, num_samples=24, hard_filepath=data_path+'/', 
                            neg_type=neg_type, query_type=query_type, loss_type=loss_type
                        )
                        tdg.generate_triplet_dataset(
                            query_answer_pairs=query_answer_pairs, 
                            output_path=data_path+'/'
                        )
            
            print("Triplet dataset for BERT finetuning created")

        else:
            raise ValueError("Dataset not selected")
        
    print("PROGRAM FINISHED")