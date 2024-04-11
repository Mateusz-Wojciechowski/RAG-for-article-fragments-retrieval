from text_chunking import split_text


class CustomTextSplitter:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def split_documents(self, texts):
        split_docs = []
        for text in texts:
            split_docs.extend(split_text(text, self.threshold))
        return split_docs
