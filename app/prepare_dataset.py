import argparse
from pathlib import Path
from shared import *

from transformation import create_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Prepare dataset',
                    description='Create a dataset for model training',
                    epilog='Jose Angel Perez Garrido - 2023')
    parser.add_argument("-o", "--option", type=str, help="select an option: create_dataset -> Parse a FAQ dataset; change_query_type -> Change query_type of all pairs; append_test_data -> Append test data to dataset", required=True)
    parser.add_argument("-q", "--query_type", type=str, help="select a query_type", default='user_query')
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalFAQ")

    args = parser.parse_args()

    path = Path.cwd()
    
    # Parse a FAQ dataset
    if args.option == "create_dataset":
        # python app\prepare_dataset.py -o create_dataset -d MentalFAQ
        if not args.dataset == "":
            file_path = F"{str(path)}/file/datasets/Mental_Health_FAQ.csv"
            posts_path = F"{str(path)}/file/datasets/Reddit_posts.csv"
            comments_path = F"{str(path)}/file/datasets/Reddit_comments.csv"
            output_path = F"{str(path)}/file/data/"+args.dataset

            query_answer_pairs = create_dataset(file_path, posts_path, comments_path)

            # Dump data to json file
            make_dirs(output_path)
            dump_to_json(query_answer_pairs, output_path+'/query_answer_pairs.json', sort_keys=False)
            print("Dataset",args.dataset, "created.")

        else:
            raise ValueError("Dataset not selected")

    # Change query_type of all pairs
    if args.option == "change_query_type":
        # python app\prepare_dataset.py -o change_query_type -d MentalFAQ -q user_query
        if not args.dataset == "":
            dataset_path = F"{str(path)}/file/data/"+args.dataset+"/query_answer_pairs.json"
        
            data = load_from_json(dataset_path)
            query_type = args.query_type

            for item in data:
                item.update({"query_type": str(query_type)})

            dump_to_json(data, dataset_path, sort_keys=False)
            print("Query_type changed to:",dataset_path)
        else:
            raise ValueError("Dataset not selected")

    # Append test data to dataset
    if args.option == "append_test_data":
        # python app\prepare_dataset.py -o append_test_data -d MentalFAQ -q user_query
        if not args.dataset == "":
            dataset_path = F"{str(path)}/file/data/"+args.dataset+"/query_answer_pairs.json"
            data_filepath = F"{str(path)}/file/test/"+args.dataset+"/test.json"
        
            dataset = load_from_json(dataset_path)
            data = load_from_json(data_filepath)

            query_type = args.query_type
            for item in data:
                item["query_type"] = str(query_type)

            dump_to_json(dataset+data, dataset_path, sort_keys=False)
            print("Test data appended to:",dataset_path)
        else:
            raise ValueError("Dataset not selected")

    print("PROGRAM FINISHED")

    