import argparse
from pathlib import Path
from shared import *
from sentence_transformers import SentenceTransformer
from semantic_search.semantic_search import semantic_search
from semantic_search.BERT_FAQ import FAQ_BERT_Finetuning
import pandas as pd

def get_finetune_results(data_path,output_path,model_name, epochs,batch_size,learning_rate):
    # define output path to save model & evaluation files
    make_dirs(output_path)

    # Creating Empty DataFrame
    df_triplet = pd.DataFrame()
    df_softmax = pd.DataFrame()

    # define variables
    for query_type in ["faq"]:
        for neg_type in ["simple"]: 
            for loss_type in ["softmax"]: 
                
                # load dataset.csv
                dataset = pd.read_csv(data_path+'/dataset/'+loss_type+'/'+query_type+'/'+neg_type+'_'+query_type+'_dataset.csv')

                # create instance of FAQ_BERT_Finetuning
                bert = FAQ_BERT_Finetuning(
                    loss_type=loss_type, query_type=query_type, neg_type=neg_type, version="1.1", epochs=epochs, batch_size=batch_size, 
                    pre_trained_name=model_name, 
                    evaluation_steps=1000, test_size=0.2, learning_rate=learning_rate
                )

                # create model and save to output path
                bert.create_model_eval(dataset, output_path)

                # load evaluation results
                if(loss_type == "triplet"):
                    temp = pd.read_csv(output_path+"/models/"+loss_type+"_"+neg_type+"_"+query_type+"_1.1/triplet_evaluation_test_results.csv")
                elif(loss_type == "softmax"):
                    temp = pd.read_csv(output_path+"/models/"+loss_type+"_"+neg_type+"_"+query_type+"_1.1/CEBinaryClassificationEvaluator_test_results.csv")

                temp.insert(0, "Loss_Type", loss_type)
                temp.insert(0, "Neg_Type", neg_type)
                temp.insert(0, "Learning_Rate", learning_rate)
                temp.insert(0, "Batch_Value", batch_size)
                temp.insert(0, "Epoch_Value", epochs)
                temp.insert(0, "Bert_model", model_name)
                temp = temp.drop(columns=['epoch', 'steps'])
                
                if loss_type == "triplet":
                    # concatenating df and temp along rows
                    df_triplet = pd.concat([df_triplet, temp], axis=0)
                elif loss_type == "softmax":
                    # concatenating df and temp along rows
                    df_softmax = pd.concat([df_softmax, temp], axis=0)

                # remove model folder
                remove_dirs(output_path+"/models")

    return {"triplet":df_triplet,"softmax":df_softmax}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Training',
                    description='Train a pretrained BERT model',
                    epilog='Jose Angel Perez Garrido - 2023')
    parser.add_argument("-o", "--option", type=str, help="select an option: model_training -> Finetune BERT model; hyperparameter_tuning -> Create CSV with results from several hyperparameter configurations (grid search); test_similarity -> Test semantic search of a SentenceTransformer model with corpus sentences", required=True)
    parser.add_argument("-m", "--model", type=str, help="select a model", default='distilbert-base-uncased')
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalFAQ")
    parser.add_argument("-e", "--epochs", type=int, help="select a number of epochs for training")
    parser.add_argument("-b", "--batch_size", type=int, help="select a batch size value")
    parser.add_argument("-lr", "--learning_rate", type=float, help="select a learning rate value")
    #parser.add_argument("-neg", "--neg_type", type=str, help="select a negative type (simple or hard)")
    #parser.add_argument("-loss", "--loss_type", type=str, help="select a loss_type (triplet or softmax)")
    parser.add_argument("-db", "--database", type=str, help="select a database name")

    args = parser.parse_args()

    path = Path.cwd()

    # Finetune BERT model
    if args.option == "model_training":
        # python app\train.py -o model_training -d MentalFAQ -m publichealthsurveillance/PHS-BERT -e 2 -b 32 -lr 3e-05
        if not args.dataset == "":
            data_path = F"{str(path)}/file/data/"+args.dataset
            output_path = F"{str(path)}/output/"+args.dataset

            model_name = args.model
            epoch_value = args.epochs
            batch_value = args.batch_size
            lr_value = args.learning_rate
            neg_type = "simple"
            loss_type = "softmax"
            query_type = "faq"

            model_output_path = output_path+"_"+model_name

            # define output path to save model & evaluation files
            make_dirs(output_path)

            # load dataset.csv
            dataset = pd.read_csv(data_path+'/dataset/'+loss_type+'/'+query_type+'/'+neg_type+'_'+query_type+'_dataset.csv')

            # create instance of FAQ_BERT_Finetuning
            bert = FAQ_BERT_Finetuning(
                loss_type=loss_type, query_type=query_type, neg_type=neg_type, version="1.1", epochs=epoch_value, batch_size=batch_value, 
                pre_trained_name=model_name, 
                evaluation_steps=1000, test_size=0.1, learning_rate=lr_value
            )

            # create model and save to output path
            bert.create_model(dataset, output_path)

            print("Model saved to", output_path)

        else:
            raise ValueError("Dataset not selected")

    # Create CSV with results from several hyperparameter configurations
    if args.option == "hyperparameter_tuning":
        # python app\train.py -o hyperparameter_tuning -d MentalFAQ -m bert-base-uncased,mental/mental-bert-base-uncased,mental/mental-roberta-base,publichealthsurveillance/PHS-BERT 
        if not args.dataset == "":
            data_path = F"{str(path)}/file/data/"+args.dataset
            output_path = F"{str(path)}/output/"+args.dataset
        
            # Test parameters
            # (from Appendix A.3 of the BERT paper)
            batch_sizes = [16, 32]
            learning_rate = [5e-5, 3e-5, 2e-5]
            adam_epsilons = [1e-8]
            epochs = [2, 3, 4]

            model_list = [str(item) for item in args.model.split(',')]

            # Creating Empty DataFrame
            df_triplet = pd.DataFrame()
            df_softmax = pd.DataFrame()

            for model_name in model_list:
                for epoch_value in epochs:
                    for batch_value in batch_sizes:
                        for lr_value in learning_rate:
                            model_output_path = output_path

                            result = get_finetune_results(data_path,model_output_path,model_name,epoch_value,batch_value,lr_value)

                            df_triplet = pd.concat([df_triplet, result["triplet"]], axis=0)
                            df_softmax = pd.concat([df_softmax, result["softmax"]], axis=0)

            # Dump data to csv file
            make_dirs(output_path+"/evaluation_results")
            df_triplet.to_csv(output_path+"/evaluation_results/triplet_results.csv")
            df_softmax.to_csv(output_path+"/evaluation_results/softmax_results.csv")

            print("Results saved to", output_path+"/evaluation_results")

        else:
            raise ValueError("Dataset not selected")

    # Test semantic search of a SentenceTransformer model with corpus sentences
    if args.option == "test_similarity":
        # python app\train.py -o test_similarity -d MentalFAQ -m <<model_name>>
        # Example: app\train.py -o test_similarity -d MentalFAQ -m all-MiniLM-L6-v2
        if not args.dataset == "":
            dataset_path = F"{str(path)}/file/data/"+args.dataset+"/query_answer_pairs.json"

            corpus_sentences = load_from_json(dataset_path)

            model = SentenceTransformer(args.model) # load pretrained model

            semantic_search(corpus_sentences,model)
        else:
            raise ValueError("Dataset not selected")

    print("PROGRAM FINISHED")