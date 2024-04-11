from langchain_community.vectorstores import FAISS
from functionalities import create_embedding_model


def retrieve_information(query, embedding_model, db_loaded):
    embedded_query = embedding_model.embed_query(query)
    search_docs = db_loaded.similarity_search_by_vector(embedded_query)
    return search_docs[0].page_content


if __name__ == '__main__':
    # creating the embedding model
    embedding_model = create_embedding_model()

    # loading the db and retrieving information
    db_loaded = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    query = "backpropagating the error in a neural network"
    print(retrieve_information(query, embedding_model, db_loaded))
