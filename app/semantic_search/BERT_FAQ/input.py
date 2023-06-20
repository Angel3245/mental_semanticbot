from elasticsearch import TransportError
from elasticsearch_dsl.connections import connections

from faq_bert_ranker import FAQ_BERT_Ranker

def ranker(top_k, model_path, fields, index_name, loss_type, neg_type, query_type, question):
    try:
        es = connections.create_connection(hosts=['localhost'])
    except TransportError as e:
        e.info()

    version = '1.1'

    model_name = "{}_{}_{}_{}".format(loss_type, neg_type, query_type, version)
    bert_model_path = "output" + "/" + model_path + "/models/" + model_name

    faq_bert_ranker = FAQ_BERT_Ranker(
        es=es, index=index_name, fields=fields, top_k=top_k, bert_model_path=bert_model_path
    )

    return faq_bert_ranker.rank_results(question)

def confidence_estimator(sentence, score):
    if(score < 8):
        return "I am not able to answer that question"
    else:
        return sentence
