from elasticsearch import helpers
from tqdm import tqdm
import logging

from model import QA

def ingest_data(data, es, index):
    """ Ingest data as a bulk of documents to ES index """

    try:
        docs = []
        for pair in tqdm(data):
            
            # initialize QA document
            doc = QA()

            if 'id' in pair:
                doc.id = pair['id']
            if 'question' in pair:
                doc.question = pair['question']
            if 'answer' in pair:
                doc.answer = pair['answer']
            if 'question' in pair and 'answer' in pair:
                doc.question_answer = pair['question'] + " " + pair['answer']

            docs.append(doc.to_dict(include_meta=False))

        # bulk indexing
        response = helpers.bulk(es, actions=docs, index=index, doc_type='doc')
            
    except Exception:
        logging.error('exception occured', exc_info=True)