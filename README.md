# EchoReads-Explainable-Book-Recommendations

EchoReads is a content-based book recommender that combines TF–IDF similarity with MMR re-ranking, author/series boosting, genre overlap, and popularity signals. It supports robust CSV/ZIP import, fuzzy title matching, language preference, and returns short explanations for each recommendation.

*Highlights*
- TF–IDF + cosine similarity for base relevance
- MMR re-ranking to increase diversity and avoid duplicates
- Author/series boosts and genre overlap weighting
- Popularity (avg rating × log(num_ratings)) nudges popular books
- Robust CSV parsing and fuzzy title matching for messy datasets
