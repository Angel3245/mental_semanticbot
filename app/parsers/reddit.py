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

def clean_answer(sentence):
    """ Apply input sentence some transformations to get a clean answer sentence:
        #1. Remove NAN
        #2. Remove weblinks
        #3. Expand contractions
    """
    sentence=contractions.fix(sentence)
    sentence=str(sentence)
    sentence=sentence.replace('{html}',"")
    rem_url=re.sub(r'http\S+', '',sentence)
    rx = re.compile(r'([^\W\d_])\1{2,}')
    rem_num = re.sub(r'[^\W\d_]+', lambda x: Word(rx.sub(r'\1\1', x.group())).correct() if rx.search(x.group()) else x.group(), rem_url)

    return rem_num

class Reddit_Parser(object):
    """ Class for parsing & extracting data from Reddit_comments.csv & Reddit_posts.csv """

    def __init__(self, posts_path, comments_path):
        # read data as pandas DataFrame
        posts_df = pd.read_csv(posts_path)
        comments_df = pd.read_csv(comments_path)

        df = self.prepare_data(posts_df, comments_df)

        self.faq_pairs = []
        self.num_faq_pairs = 0

        self.extract_data(df)

    def filter_irrelevant_posts(self, posts_df):
        # filter by flair
        flairs = ["DAE Questions", "Question", ":snoo_thoughtful: help? :snoo_biblethump:",
                ":orly: Help please!", "DAE?", "Advice", "Advice Needed","REQUESTING ADVICE"]
        posts_df = posts_df.apply(lambda row: row[posts_df['link_flair_text'].isin(flairs)])

        # check if title contains a question mark
        posts_df = posts_df[posts_df['title'].str.contains('\?')]

        # create a new title_length column that contains the number of words per title:
        posts_df["title_length"] = posts_df.apply(
            lambda x: len(x["title"].split()), axis=1
        )
        # remove short posts
        posts_df = posts_df[posts_df['title_length'] > 4]

        return posts_df

    def filter_irrelevant_comments(self, comments_df):
        # create a new comments_length column that contains the number of words per comment:
        comments_df["comment_length"] = comments_df.apply(
            lambda x: len(x["body"].split()), axis=1
        )

        # filter out short comments, which typically include things “Thanks!” that are not relevant for our search engine.
        comments_df = comments_df[comments_df["comment_length"] > 10]

        # remove answers with low scores
        comments_df = comments_df[comments_df['score'] > 3]

        # get answer with best score
        comments_df = comments_df.sort_values('score', ascending=False).drop_duplicates(['parent_id'])

        return comments_df
        
    def prepare_data(self, posts_df, comments_df):
        # drop null values
        posts_df.dropna(inplace=True)
        comments_df.dropna(inplace=True)

        #filter posts columns
        columns = posts_df.columns
        columns_to_keep = ["title", "score", "name", "link_flair_text"]
        columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
        posts_df = posts_df.drop(columns_to_remove, axis=1)
        # rename colnames body: query_string, score: post_score
        posts_df = posts_df.rename(columns={'score': 'post_score'})

        # filter comments columns
        columns = comments_df.columns
        columns_to_keep = ["body", "score", "parent_id"]
        columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
        comments_df = comments_df.drop(columns_to_remove, axis=1)
        
        # filter irrelevant data
        posts_df = self.filter_irrelevant_posts(posts_df)
        comments_df = self.filter_irrelevant_comments(comments_df)
        
        # truncate too long texts to avoid memory overload
        comments_df['body'] = comments_df['body'].str.split(n=100).str[:100].str.join(' ')

        # join both dataframes
        df = posts_df.merge(comments_df,left_on='name', right_on='parent_id')

        # rename title column to question
        df = df.rename(columns={'title': 'question'})

        # rename colname body: answer
        df = df.rename(columns={'body': 'answer'})

        # clean text
        df['question']=df['question'].map(lambda s:clean_question(s)) 
        df['answer']=df['answer'].map(lambda s:clean_answer(s))

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
                data["label"] = 1
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
        """ Generate query answer pair list using faq, user query pairs 
        
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

        