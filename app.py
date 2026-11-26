
from pathlib import Path
import streamlit as st
import pandas as pd

from search_engine import EducationalSearchEngine

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EduSearch AI",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "educational_documents.csv"

if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "educational_documents.csv"

RESULTS_DIR = BASE_DIR / "results"

# --------------------------------------------------
# LOAD NLP SEARCH ENGINE
# --------------------------------------------------

@st.cache_resource
def load_engine():
    return EducationalSearchEngine(DATA_PATH)

engine = load_engine()

documents = engine.df

categories = sorted(
    documents["category"].unique().tolist()
)

# --------------------------------------------------
# APPLICATION STYLING
# --------------------------------------------------

st.markdown("""
<style>

.hero {
    background: linear-gradient(
        120deg, #10233e, #1d4ed8
    );
    padding: 32px;
    border-radius: 16px;
    color: white;
    margin-bottom: 24px;
}

.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 4px;
}

.hero p {
    color: #e2e8f0;
    font-size: 17px;
    margin-bottom: 8px;
}

.eyebrow {
    font-size: 12px;
    letter-spacing: 2px;
    color: #bfdbfe;
}

div[data-testid="stMetric"] {
    border: 1px solid #94a3b855;
    padding: 16px;
    border-radius: 12px;
}

</style>

<div class="hero">
    <div class="eyebrow">
        NIELIT AIDA | NLP PROJECT
    </div>
    <h1>EduSearch AI</h1>
    <p>Smart Educational Document Search Engine</p>
    <div>
        Discover Knowledge. Search Smarter.
    </div>
    <br>
    Powered by TF-IDF and Cosine Similarity
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# PROJECT METRICS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Educational Documents",
    len(documents)
)

col2.metric(
    "Subject Categories",
    len(categories)
)

col3.metric(
    "Vocabulary Features",
    f"{len(engine.feature_names):,}"
)

col4.metric(
    "Search Limit",
    "Top 5"
)

st.divider()

# --------------------------------------------------
# APPLICATION TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Document Search",
    "📚 Document Library",
    "📊 Technical Evaluation",
    "🎓 About Project"
])

# ==================================================
# TAB 1: DOCUMENT SEARCH
# ==================================================

with tab1:

    st.header("Search Educational Documents")

    st.caption(
        "Enter a question or keywords to discover "
        "relevant educational notes."
    )

    with st.form("search_form"):

        query = st.text_input(
            "Enter your educational query",
            value="Explain neural networks used in deep learning.",
            key="search_query"
        )

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:

            selected_category = st.selectbox(
                "Subject Category",
                ["All Categories"] + categories,
                key="search_category"
            )

        with filter_col2:

            min_score = st.slider(
                "Minimum Similarity Score",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                key="search_threshold"
            )

        submitted = st.form_submit_button(
            "🔎 Search Documents",
            type="primary"
        )

    if submitted:

        st.session_state["search_request"] = (
            query,
            selected_category,
            min_score
        )

    request = st.session_state.get("search_request")

    if request is not None:

        active_query, active_category, active_score = request

        if not active_query.strip():

            st.warning("Please enter a valid search query.")

        else:

            category_filter = (
                None
                if active_category == "All Categories"
                else active_category
            )

            results = engine.search(
                active_query,
                top_n=5,
                min_score=active_score,
                category=category_filter
            )

            if results.empty:

                st.warning(
                    "No matching documents found. "
                    "Try different keywords or reduce "
                    "the similarity threshold."
                )

            else:

                st.success(
                    f"{len(results)} relevant document(s) found."
                )

                st.caption(f"Search query: {active_query}")

                st.subheader("Ranked Search Results")

                for _, row in results.iterrows():

                    with st.container(border=True):

                        st.caption(
                            f"RANK {row['rank']} | "
                            f"{row['document_id']}"
                        )

                        st.subheader(row["title"])

                        st.caption(row["category"])

                        score = float(row["similarity_score"])

                        st.markdown(
                            f"**Cosine Similarity:** {score:.6f}"
                        )

                        st.progress(
                            min(1.0, max(0.0, score))
                        )

                        st.write(row["snippet"])

                        with st.expander("Read Full Document"):

                            document = documents.loc[
                                documents["document_id"]
                                == row["document_id"]
                            ].iloc[0]

                            st.write(document["content"])

                            st.caption(
                                f"Source: {document['source']}"
                            )

                st.subheader("Similarity Comparison")

                chart_data = results[
                    ["title", "similarity_score"]
                ].set_index("title")

                st.bar_chart(chart_data)

                st.subheader("Search Result Summary")

                st.dataframe(
                    results[
                        [
                            "rank",
                            "document_id",
                            "title",
                            "similarity_score"
                        ]
                    ],
                    hide_index=True
                )

                st.download_button(
                    "Download Search Results",
                    data=results.to_csv(index=False),
                    file_name="search_results.csv",
                    mime="text/csv"
                )

    else:

        st.info(
            "Enter a query and click Search Documents."
        )

    st.caption(
        "Similarity scores indicate textual matching, "
        "not factual accuracy or probability."
    )

# ==================================================
# TAB 2: DOCUMENT LIBRARY
# ==================================================

with tab2:

    st.header("Educational Document Library")

    library_category = st.selectbox(
        "Browse by Subject",
        ["All Categories"] + categories,
        key="library_category"
    )

    library = documents.copy()

    if library_category != "All Categories":

        library = library[
            library["category"] == library_category
        ]

    st.caption(
        f"Displaying {len(library)} educational documents."
    )

    for _, document in library.iterrows():

        with st.expander(
            f"{document['document_id']} - "
            f"{document['title']}"
        ):

            st.caption(document["category"])

            st.write(document["content"])

            st.caption(
                f"Source: {document['source']}"
            )

# ==================================================
# TAB 3: TECHNICAL EVALUATION
# ==================================================

with tab3:

    st.header("NLP Technical Evaluation")

    tech1, tech2, tech3 = st.columns(3)

    tech1.metric(
        "Document Vectors",
        engine.tfidf_matrix.shape[0]
    )

    tech2.metric(
        "Vocabulary Size",
        engine.tfidf_matrix.shape[1]
    )

    tech3.metric(
        "Non-Zero Values",
        engine.tfidf_matrix.nnz
    )

    st.subheader("TF-IDF Configuration")

    configuration = pd.DataFrame({
        "Parameter": [
            "Vectorizer",
            "Stop Words",
            "N-gram Range",
            "Similarity Method",
            "Matrix Dimensions",
            "Ranking Method"
        ],
        "Value": [
            "TfidfVectorizer",
            "English",
            "Unigrams and Bigrams (1, 2)",
            "Cosine Similarity",
            str(engine.tfidf_matrix.shape),
            "Descending Similarity Score"
        ]
    })

    st.dataframe(
        configuration,
        hide_index=True
    )

    st.subheader("TF-IDF Vocabulary")

    vocabulary = pd.DataFrame({
        "Feature": engine.feature_names,
        "IDF Weight": engine.vectorizer.idf_
    })

    st.dataframe(
        vocabulary,
        hide_index=True
    )

    st.download_button(
        "Download TF-IDF Vocabulary",
        data=vocabulary.to_csv(index=False),
        file_name="tfidf_vocabulary.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("Systematic Testing Results")

    test_path = RESULTS_DIR / "test_queries.csv"

    if test_path.exists():

        tests = pd.read_csv(test_path)

        t1, t2, t3, t4 = st.columns(4)

        t1.metric(
            "Total Tests",
            len(tests)
        )

        t2.metric(
            "Passed",
            int((tests["status"] == "PASS").sum())
        )

        t3.metric(
            "Partial",
            int((tests["status"] == "PARTIAL").sum())
        )

        t4.metric(
            "Failed",
            int((tests["status"] == "FAIL").sum())
        )

        st.dataframe(
            tests,
            hide_index=True
        )

        st.caption(
            "Results are based on project-specific "
            "test cases and manually selected "
            "relevance labels."
        )

    else:

        st.info("Testing results file is not available.")

    graph_path = (
        RESULTS_DIR / "query_similarity_comparison.png"
    )

    if graph_path.exists():

        st.image(
            str(graph_path),
            caption="Highest cosine similarity across test queries."
        )

# ==================================================
# TAB 4: ABOUT PROJECT
# ==================================================

with tab4:

    st.header("About the Project")

    st.markdown("""
    ### NIELIT AIDA – NLP Practical Project

    **Project Title:** Smart Educational Document Search
    Engine Using TF-IDF and Cosine Similarity

    **Problem Statement:** Problem 5 –
    Educational Document Search Engine

    **Student:** Madhav Raman

    **Enrollment Number:** 2001191

    **Project Type:** Individual Project

    ---

    ### Project Objective

    To retrieve relevant educational notes from a
    structured document collection using natural
    language search queries.

    ### Methodology

    1. Educational dataset preparation.
    2. Text preprocessing.
    3. TF-IDF feature extraction.
    4. Query vectorization.
    5. Cosine similarity calculation.
    6. Document ranking and retrieval.
    7. Testing and evaluation.

    ### Dataset Source

    Original AI-assisted educational notes
    prepared specifically for this project.

    ### Limitations

    - Depends on overlapping vocabulary.
    - Does not inherently understand synonyms.
    - Does not capture deeper semantic context.
    - Similarity scores do not guarantee relevance.
    - Retrieval quality depends on the document corpus.
    """)

    st.subheader("Cosine Similarity Formula")

    st.latex(
        r"\text{Similarity}(A,B)="
        r"\frac{A\cdot B}{\|A\|\|B\|}"
    )

    st.markdown("""
    ### Technologies Used

    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF
    - Cosine Similarity
    - Matplotlib
    - Streamlit
    """)

st.divider()

st.caption(
    "EduSearch AI | NIELIT AIDA Individual NLP Project "
    "| Academic and Educational Use"
)
