# RAG-for-article-fragments-retrieval
# Checking the results:
To check model's performance please navigate to model_performance.ipynb file, you will find a few responses from the model to sample queries

# Setting up the project:
I have provided a FAISS db in the repository, but in case you want to create one yourself, run the create_db.py file. A new vector db will be then created. It may take a while for a larger dataset
The chunker I implemented is located in text_chunking.py, CustomSplitter class is provided to use chunking along with langchain.
To test the performance of the model yourself go to retrieve_information.py, provide a query and run the file, you will receive a relevant fragment of an article from the model.
There is also a functionalities.py file with a method responsible for creating an embedding model.
The requirements.txt contains all the requirements, medium.csv and medium_cp1250_compatible.csv are both files containing the dataset, in different encodings, I have change the encoding to cp1250 for compatibility with langchain library, code for that purpose is provided in data_preprocessing.ipynb. I have obtained the dataset from following source: https://www.kaggle.com/datasets/meruvulikith/1300-towards-datascience-medium-articles-dataset
