import math
from collections import Counter

class BM25Lite:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        self.N = len(corpus)

        self.df = {}
        for doc in corpus:
            for word in set(doc):
                self.df[word] = self.df.get(word, 0) + 1

        self.idf = {word: math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                    for word, df in self.df.items()}

        self.tf = [Counter(doc) for doc in corpus]

    def get_scores(self, query):
        query_words = query
        scores = []

        for idx, doc in enumerate(self.corpus):
            score = 0
            doc_tf = self.tf[idx]
            dl = self.doc_len[idx]

            for word in query_words:
                if word in doc_tf:
                    tf = doc_tf[word]
                    idf = self.idf.get(word, 0)
                    score += idf * (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    )

            scores.append(score)

        return scores

