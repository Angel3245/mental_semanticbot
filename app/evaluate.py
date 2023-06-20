import argparse
from pathlib import Path
from elasticsearch import Elasticsearch
from shared import *

from semantic_search.BERT_FAQ import ReRanker, get_relevance_label_df, Evaluation

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Evaluate',
                    description='Evaluate the results of the semantic search system',
                    epilog='Jose Angel Perez Garrido - 2023')
    parser.add_argument("-o", "--option", type=str, help="select an option: generate_topk -> Generate ES top-k results for re-ranking; generate_BERT_results -> Generate BERT prediction results; generate_reranked_results -> Perform re-ranking; evaluation -> Generate evaluation metrics: NDCG@1, NDCG@3, P@1, P@3, MAP", required=True)
    parser.add_argument("-m", "--model", type=str, help="select a model", default='distilbert-base-uncased')
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalFAQ")

    args = parser.parse_args()

    path = Path.cwd()

    # Generate ES top-k results for re-ranking
    if args.option == "generate_topk":
        # python app\evaluate.py -o generate_topk -d MentalFAQ
        if not args.dataset == "":
            data_path = F"{str(path)}/file/data/"+args.dataset

            # Generate list of test queries, relevance labels for ReRanker class
            query_answer_pair_filepath = data_path+'/query_answer_pairs.json'
            relevance_label_df = get_relevance_label_df(query_answer_pair_filepath)
            test_queries = relevance_label_df[relevance_label_df['query_type'] == 'user_query'].question.unique()

            test_queries = list(test_queries)

            # Define instance of ReRanker class
            r = ReRanker(
                bert_model_path='', 
                test_queries=test_queries, relevance_label_df=relevance_label_df
            )

            # create output path to save Elasticsearch top-k results
            output_path = data_path+"/rank_results/unsupervised"
            make_dirs(output_path)

            # Get top-k Elasticsearch results 
            es = Elasticsearch([{'host':'localhost','port':9200}], http_auth=('elastic', 'elastic')) 
            index_name = args.dataset.lower()
            top_k = 100

            es_query_by_question = r.get_es_topk_results(es=es, index=index_name, query_by=['question'], top_k=top_k)
            es_query_by_answer = r.get_es_topk_results(es=es, index=index_name, query_by=['answer'], top_k=top_k)
            es_query_by_question_answer = r.get_es_topk_results(es=es, index=index_name, query_by=['question', 'answer'], top_k=top_k)
            es_query_by_question_answer_concat = r.get_es_topk_results(es=es, index=index_name, query_by=['question_answer'], top_k=top_k)

            # Save Elasticsearch results to json files
            dump_to_json(es_query_by_question, output_path + '/es_query_by_question.json')
            dump_to_json(es_query_by_answer, output_path + '/es_query_by_answer.json')
            dump_to_json(es_query_by_question_answer, output_path + '/es_query_by_question_answer.json')
            dump_to_json(es_query_by_question_answer_concat, output_path + '/es_query_by_question_answer_concat.json')

            print("ElasticSearch top-k results for re-ranking generated to",output_path)

        else:
            raise ValueError("Dataset not selected")
            
    # Generate BERT prediction results
    if args.option == "generate_BERT_results":
        # python app\evaluate.py -o generate_BERT_results -d MentalFAQ 
        
        if not args.dataset == "":
            data_path = F"{str(path)}/file/data/"+args.dataset
            model_path = F"{str(path)}/output/"+args.dataset+"/models"

            output_path=data_path+"/rank_results"

            # load user_query ES results from json files
            es_output_path = output_path + "/unsupervised"
            es_query_by_question = load_from_json(es_output_path + '/es_query_by_question.json')
            es_query_by_answer = load_from_json(es_output_path + '/es_query_by_answer.json')
            es_query_by_question_answer = load_from_json(es_output_path + '/es_query_by_question_answer.json')
            es_query_by_question_answer_concat = load_from_json(es_output_path + '/es_query_by_question_answer_concat.json')

            # load test_queries, relevance_label_df for ReRanker
            query_answer_pair_filepath = data_path+'/query_answer_pairs.json'
            relevance_label_df = get_relevance_label_df(query_answer_pair_filepath)
            test_queries = relevance_label_df[relevance_label_df['query_type'] == 'user_query'].question.unique()

            version="1.1"
            # define rank_field parameter
            for rank_field in ["BERT-Q-a","BERT-Q-q"]: 

                # define variables
                for query_type in ["faq"]:
                    for neg_type in ["simple"]:
                        for loss_type in ["softmax"]:
                            bert_model_path=model_path + '/' + loss_type + '_' + neg_type + '_' + query_type + '_' + version

                            # create instance of ReRanker class
                            r = ReRanker(
                                bert_model_path=bert_model_path, 
                                test_queries=test_queries, relevance_label_df=relevance_label_df,
                                rank_field=rank_field
                            )

                            # generate directory structure
                            pred_output_path = output_path + "/supervised/" + rank_field + "/" + loss_type + "/" + query_type + "/" + neg_type
                            make_dirs(pred_output_path)

                            # next, generate BERT, Re-ranked top-k results and dump to files
                            bert_query_by_question = r.get_bert_topk_preds(es_query_by_question)
                            dump_to_json(bert_query_by_question, pred_output_path + '/bert_query_by_question.json')

                            bert_query_by_answer = r.get_bert_topk_preds(es_query_by_answer)
                            dump_to_json(bert_query_by_answer, pred_output_path + '/bert_query_by_answer.json')

                            bert_query_by_question_answer = r.get_bert_topk_preds(es_query_by_question_answer)
                            dump_to_json(bert_query_by_question_answer, pred_output_path + '/bert_query_by_question_answer.json')

                            bert_query_by_question_answer_concat = r.get_bert_topk_preds(es_query_by_question_answer_concat)
                            dump_to_json(bert_query_by_question_answer_concat, pred_output_path + '/bert_query_by_question_answer_concat.json')

            print("BERT prediction results generated to",pred_output_path)
        else:
            raise ValueError("Dataset not selected")

    # Perform re-ranking
    if args.option == "generate_reranked_results":
        # python app\evaluate.py -o generate_reranked_results -d MentalFAQ
        if not args.dataset == "":
            data_path = F"{str(path)}/file/data/"+args.dataset

            # define output path
            output_path=data_path+"/rank_results"

            # define rank_field, w_t parameters
            w_t=10
            for rank_field in ["BERT-Q-a","BERT-Q-q"]: 

                # define variables
                for query_type in ["faq"]:
                    for neg_type in ["simple"]:
                        for loss_type in ["softmax"]:
                            # create instance of ReRanker class
                            r = ReRanker(rank_field=rank_field, w_t=w_t)

                            reranked_output_path = output_path + "/supervised/" + rank_field + "/" + loss_type + "/" + query_type + "/" + neg_type
                            pred_output_path     = output_path + "/supervised/" + rank_field + "/" + loss_type + "/" + query_type + "/" + neg_type

                            # generate reranked results 
                            bert_query_by_question = load_from_json(pred_output_path + '/bert_query_by_question.json')
                            reranked_query_by_question = r.get_reranked_results(bert_query_by_question)
                            dump_to_json(reranked_query_by_question, reranked_output_path + '/reranked_query_by_question.json')

                            bert_query_by_answer = load_from_json(pred_output_path + '/bert_query_by_answer.json')
                            reranked_query_by_answer = r.get_reranked_results(bert_query_by_answer)
                            dump_to_json(reranked_query_by_answer, reranked_output_path + '/reranked_query_by_answer.json')

                            bert_query_by_question_answer = load_from_json(pred_output_path + '/bert_query_by_question_answer.json')
                            reranked_query_by_question_answer = r.get_reranked_results(bert_query_by_question_answer)
                            dump_to_json(reranked_query_by_question_answer, reranked_output_path + '/reranked_query_by_question_answer.json')

                            bert_query_by_question_answer_concat = load_from_json(pred_output_path + '/bert_query_by_question_answer_concat.json')
                            reranked_query_by_question_answer_concat = r.get_reranked_results(bert_query_by_question_answer_concat)
                            dump_to_json(reranked_query_by_question_answer_concat, reranked_output_path + '/reranked_query_by_question_answer_concat.json')

            print("Re-ranking results generated to",reranked_output_path)
        else:
            print("Dataset not selected")

    # Generate evaluation metrics: NDCG@1, NDCG@3, P@1, P@3, MAP
    if args.option == "evaluation":
        # python app\evaluate.py -o evaluation -d MentalFAQ
        if not args.dataset == "":
            data_path = F"{str(path)}/file/data/"+args.dataset
            output_path = F"{str(path)}/output/"+args.dataset

            query_answer_pairs_filepath = data_path+'/query_answer_pairs.json'
            rank_results_filepath=data_path+"/rank_results"

            ev = Evaluation(qas_filename=query_answer_pairs_filepath,rank_results_filepath=rank_results_filepath, loss_types=["softmax"], query_types=["faq"], test_data="user_query", top_k=[1,3])
            df = ev.get_eval_df()

            df.to_csv(output_path+"/model_evaluation.csv")

            print("Evaluation metrics generated to",output_path)

        else:
            raise ValueError("Dataset not selected")