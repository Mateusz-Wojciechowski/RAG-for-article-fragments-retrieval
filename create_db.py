from langchain_community.document_loaders.csv_loader import CSVLoader
from functionalities import create_embedding_model
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
embedding_model = create_embedding_model()

# creating and saving db
db = FAISS.from_texts(docs, embedding_model)
db.save_local("faiss_index")
