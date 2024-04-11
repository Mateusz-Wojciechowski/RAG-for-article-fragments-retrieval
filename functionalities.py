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

