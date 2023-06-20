import pandas as pd

from nltk.tokenize import RegexpTokenizer
from textblob import Word
import re
import contractions

def clean_question(sentence):
    """ Apply input sentence several transformations to get a clean question sentence:
        #1. Lowercase text
        #2. Remove whitespace
        #3. Remove numbers
        #4. Remove special characters
        #5. Remove emails
        #6. Remove text inside parentheses
        #7. Remove NAN
        #8. Remove weblinks
        #9. Expand contractions
        #10. Tokenize
    """
    sentence=contractions.fix(sentence)
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"")
    sentence=re.sub("[\(\[<].*?[\)\]>]", "", sentence)
    rem_url=re.sub(r'http\S+', '',sentence)
    rem_num = re.sub('[0-9]+', '', rem_url)
    rx = re.compile(r'([^\W\d_])\1{2,}')
    rem_num = re.sub(r'[^\W\d_]+', lambda x: Word(rx.sub(r'\1\1', x.group())).correct() if rx.search(x.group()) else x.group(), rem_num)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)

    return " ".join(tokens)

class MentalFAQ_Parser(object):
    """ Class for parsing & extracting data from Mental_Health_FAQ.csv """

    def __init__(self, file_path):
        # read data as pandas DataFrame
        df = pd.read_csv(file_path)

        df = self.prepare_data(df)

        self.faq_pairs = []
        self.num_faq_pairs = 0

        self.extract_data(df)
        
    def prepare_data(self, df):
        # rename colnames Questions: question, Answers: answer
        df = df.rename(columns={'question1': 'query_string', 'Questions': 'question','Answers':'answer'})

        # drop null values
        df.dropna(inplace=True)

        # clean text
        df['question']=df['question'].map(lambda s:clean_question(s))
        
        # truncate too long comments to avoid memory overload
        df['answer'] = df['answer'].str.split(n=100).str[:100].str.join(' ')

        return df

    def extract_pairs(self, df, query_type):
        """ Extract qa pairs from DataFrame for a given query_type
    
        :param df: input DataFrame
        :param query_type: faq or user_query
        :return: qa pairs
        """
        qa_pairs = []
        if query_type == "faq":
            # select question, answer columns
            df = df[['question', 'answer']]

            for _, row in df.iterrows():
                data = dict()
                data["query_type"] = "faq"
                data["question"] = row["question"]
                data["answer"] = row["answer"]
                qa_pairs.append(data)

        else:
            raise ValueError('error, no query_type found for {}'.format(query_type))

        # remove duplicates
        pairs = []
        for pair in qa_pairs:
            if pair not in pairs:
                pairs.append(pair)

        return pairs

    def get_query_answer_pairs(self, faq_pairs):
        """ Generate query answer pair list using faq pairs 
        
        :param faq_pairs: faq pairs
        :return: query answer pairs
        """
        query_answer_pairs = faq_pairs
        
        return query_answer_pairs

            
    def extract_data(self, df):
        """ Extract data from DataFrame
        
        :param df: Pandas DataFrame
        """
        # extract faq_pairs
        faq_pairs = self.extract_pairs(df, query_type='faq')
        
        self.faq_pairs = faq_pairs
        self.num_faq_pairs = len(faq_pairs)

        