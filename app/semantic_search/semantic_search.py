import time
from sentence_transformers import util

def semantic_search(corpus_sentences,model):
    '''Test the semantic search of a SentenceTransformer model with corpus sentences'''
    top_k = 5  # Number of passages we want to retrieve with the bi-encoder

    print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))

    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_tensor=True)

    #Move embeddings to the target device of the model
    corpus_embeddings = corpus_embeddings.to(model._target_device)

    query = input("Please enter a question (or quit): ")
    while query != "quit":
        start_time = time.time()
        question_embedding = model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
        end_time = time.time()
        hits = hits[0]  #Get the hits for the first query

        print("Input question:", query)
        print("Results (after {:.3f} seconds):".format(end_time-start_time))
        for hit in hits[0:5]:
            print("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))

        print("\n\n========\n") 

        print("BEST Answer for the given question is: ",corpus_sentences[hits[0]['corpus_id']][1])

        query = input("Please enter a question (or quit): ")