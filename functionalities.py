from langchain_community.embeddings import HuggingFaceEmbeddings


def create_embedding_model():
    model_path = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {'device': 'cpu'}

    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    return embeddings


def retrieve_fragment(retriever, embedding_model, question):
    embedded_question = embedding_model.embed_query(question)
    search_docs = retriever.get_relevant_documents(embedded_question)
    return search_docs.page_content[0]
