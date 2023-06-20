from elasticsearch_dsl import Document, Integer, Text

class QA(Document):
    id = Integer()
    question = Text()
    answer = Text()
    question_answer = Text()