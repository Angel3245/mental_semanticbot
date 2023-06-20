from parsers.reddit import clean_question
from pathlib import Path
from semantic_search.BERT_FAQ import FAQ_BERT_Ranker
from DB.connect import get_elasticsearch_connection

def get_answer(question, model_path=None, dataset='MentalFAQ', fields=['question']):
    # Clean input question
    question = clean_question(question)
    
    # Define model parameters
    top_k = 1 # one answer

    index_name = dataset.lower()

    if(model_path == None):
        # Load saved model
        path = Path.cwd()
        bert_model_path = F"{str(path)}/file/chatbot_model/softmax_PHS-BERT"
    else:
        bert_model_path = model_path

    es = get_elasticsearch_connection()

    if not es  == None:
        faq_bert_ranker = FAQ_BERT_Ranker(
            es=es, index=index_name, fields=fields, top_k=top_k, bert_model_path=bert_model_path
        )

        # Get best result
        answer = faq_bert_ranker.rank_results(question)[0]

        # Validate answer confidence        
        print("Total score: "+str(answer["score"]))
        answer = confidence_estimator(answer)

        return answer
    
    return "There was an error creating a response"

def confidence_estimator(answer):
    if(answer):
        if(answer["score"] < 8):
            return "I am not able to answer that question"
        else:
            return "Answer: "+answer["answer"]
    else:
        return "Error: Answer was not retrieved"
