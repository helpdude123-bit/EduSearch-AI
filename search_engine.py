
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Text preprocessing function
def clean_text(text):
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    return " ".join(text.split())


# Educational Document Search Engine
class EducationalSearchEngine:

    def __init__(self, csv_path):

        # Load educational dataset
        self.df = pd.read_csv(csv_path)

        # Combine document title and content
        combined = self.df["title"] + " " + self.df["content"]

        # Apply preprocessing
        self.df["cleaned_text"] = combined.apply(clean_text)

        # Initialise TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        # Fit and transform documents
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["cleaned_text"]
        )

        # Extract vocabulary
        self.feature_names = self.vectorizer.get_feature_names_out()


    # Document search function
    def search(
        self,
        query,
        top_n=5,
        min_score=0.0,
        category=None
    ):

        columns = [
            "rank",
            "document_id",
            "title",
            "category",
            "similarity_score",
            "snippet"
        ]

        # Validate user input
        if not query.strip():
            return pd.DataFrame(columns=columns)

        # Transform user query
        query_vector = self.vectorizer.transform(
            [clean_text(query)]
        )

        # Calculate cosine similarity
        scores = cosine_similarity(
            query_vector,
            self.tfidf_matrix
        ).ravel()

        # Prepare search results
        results = self.df[
            ["document_id", "title", "category", "content"]
        ].copy()

        results["similarity_score"] = scores

        # Remove zero scores and apply threshold
        results = results[
            (results["similarity_score"] > 0) &
            (results["similarity_score"] >= min_score)
        ]

        # Apply optional category filter
        if category is not None:
            results = results[
                results["category"] == category
            ]

        # Rank documents
        results = results.sort_values(
            ["similarity_score", "document_id"],
            ascending=[False, True]
        ).head(top_n).reset_index(drop=True)

        # Assign ranks
        results.insert(
            0, "rank", range(1, len(results) + 1)
        )

        # Generate document snippets
        results["snippet"] = results["content"].apply(
            lambda text: text[:250] +
            ("..." if len(text) > 250 else "")
        )

        return results[columns]
