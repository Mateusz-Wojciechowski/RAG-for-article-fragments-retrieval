from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from CustomSplitter import CustomTextSplitter

# loading the csv
loader = CSVLoader('medium_cp1250_compatible.csv', source_column='Text')
data = loader.load()

# converting to text and chunking
string_text = [data[i].page_content for i in range(len(data))]
text_splitter = CustomTextSplitter(threshold=0.6)
docs = text_splitter.split_documents(string_text)

# creating embedding model
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# creating and saving db
db = FAISS.from_texts(docs, embeddings)
db.save_local("faiss_index")
