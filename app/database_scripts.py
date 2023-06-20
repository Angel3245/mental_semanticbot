import argparse
from pathlib import Path
from transformation import *
from app.DB.connect import database_connect
from model import *
import csv
from elasticsearch_dsl import Index
from elasticsearch_dsl.connections import connections

from DB.indexer import ingest_data
from model.QA import QA
import logging

from semantic_search.BERT_FAQ import get_relevance_label_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Database scripts',
                    description='Contains methods to communicate with a DB (get and ingest data)',
                    epilog='Jose Angel Perez Garrido - 2023')
    parser.add_argument("-o", "--option", type=str, help="select an option: create_reddit_csv -> create csv datasets from Reddit data in DB; ingest_db -> insert data from /file/data/<dataset_name> into ElasticSearch" , required=True)
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalFAQ")
    parser.add_argument("-db", "--database", type=str, help="select a database name")
    
    args = parser.parse_args()

    path = Path.cwd()

    # Create csv from data in DB. It needs a connection to a Reddit DB (MySQL). (See https://github.com/gbgonzalez/reddit_extraction)
    if args.option == "create_reddit_csv":
        # python app\database_scripts.py -o create_reddit_csv -db <<database_name>>
        # Example: python app\database_scripts.py -o create_reddit_csv -db reddit
        session = database_connect(args.database)

        # Create Reddit_posts.csv
        outfile = open(F"{str(path)}/file/datasets/Reddit_posts.csv", 'w', encoding='utf-8')
        outcsv = csv.writer(outfile)
        records = session.query(Post).all()

        outcsv.writerow(['id', 'name', 'user_id', 'subreddit_id', 'permalink', 'body', 'body_html', 'title', 'created_utc', 'downs',
                    'no_follow', 'score', 'send_replies', 'stickied', 'ups', 'link_flair_text', 'link_flair_type'])
        for item in records: 
            outcsv.writerow([item.id, item.name, item.user_id, item.subreddit_id, item.permalink, item.body, item.body_html, 
                             item.title, item.created_utc, item.downs, item.no_follow, item.score, item.send_replies,
                             item.stickied, item.ups, item.link_flair_text, item.link_flair_type])

        outfile.close()

        # Create Reddit_comments.csv
        outfile = open(F"{str(path)}/file/datasets/Reddit_comments.csv", 'w', encoding='utf-8')
        outcsv = csv.writer(outfile)
        records = session.query(Comment).all()
        
        outcsv.writerow(['id', 'name', 'user_id', 'subreddit_id', 'body', 'body_html', 'created_utc', 'downs',
                    'no_follow', 'score', 'send_replies', 'stickied', 'ups', 'permalink', 'parent_id'])
        for item in records: 
            outcsv.writerow([item.id, item.name, item.user_id, item.subreddit_id, item.body, item.body_html, 
                             item.created_utc, item.downs, item.no_follow, item.score, item.send_replies,
                             item.stickied, item.ups, item.permalink, item.parent_id])

        outfile.close()

        print("Datasets Reddit_posts.csv & Reddit_comments.csv created")

    # Insert previously parsed data into a DB. It needs a connection to a DB (ElasticSearch)
    if args.option == "ingest_db":
        # python app\database_scripts.py -o ingest_db -d MentalFAQ
        if not args.dataset == "":
            data_path = F"{str(path)}/file/data/"+args.dataset

            # get list of query answer pairs
            query_answer_pairs_filepath = data_path+'/query_answer_pairs.json'
            relevance_label_df = get_relevance_label_df(query_answer_pairs_filepath)
            faq_qa_pair_df = relevance_label_df[relevance_label_df['query_type'] == 'faq']
            faq_qa_pairs = faq_qa_pair_df.T.to_dict().values()

            # convert to list
            faq_qa_pairs = list(faq_qa_pairs)

            index_name = args.dataset.lower()

            try:
                es = connections.create_connection(hosts=['localhost'])
                
                # Index data to Elasticsearch
                
                # Initialize index (only perform once)
                index = Index(index_name)

                # Define custom settings
                index.settings(
                    number_of_shards=1,
                    number_of_replicas=0
                )

                # Delete the index, ignore if it doesn't exist
                index.delete(ignore=404)

                # Create the index in Elasticsearch
                index.create()

                # Register a document with the index
                index.document(QA)

                # Ingest data to Elasticsearch
                ingest_data(faq_qa_pairs, es=es, index=index_name)

                print("Finished indexing {} records to {} index".format(len(faq_qa_pairs), index_name))

            except Exception:
                logging.error('exception occured', exc_info=True)

        else:
            raise ValueError("Dataset not selected")

    print("PROGRAM FINISHED")