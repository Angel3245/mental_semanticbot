from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from elasticsearch import Elasticsearch
from elasticsearch_dsl.connections import connections
from elasticsearch import TransportError

def database_connect(target: str = "reddit"):
        
    if(target == "reddit"):
        engine = create_engine(
            f"mysql+pymysql://redditUser:redditPass@localhost:3306/{target}?charset=utf8mb4"
        )
        Session = sessionmaker(bind=engine, autoflush=False)

    return Session()

def elasticsearch_connect():
    return Elasticsearch([{'host':'localhost','port':9200}], http_auth=('username', 'password')) 

def get_elasticsearch_connection():
    try:
        return connections.create_connection(hosts=['localhost'])
    except TransportError as e:
        e.info()
        return None