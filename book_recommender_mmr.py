# Book recommender with MMR, author/series boosts, language pref, and explanations
# Save as book_recommender_mmr.py and run (Python 3.8+)
# In Jupyter run: !pip install pandas scikit-learn numpy

import os
import zipfile
import csv
import pandas as pd
import numpy as np
import re
import math
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# 1) CSV / ZIP helpers
# ---------------------------
def detect_delimiter_and_read(filepath, max_bytes=8192):
    with open(filepath, 'rb') as f:
        sample = f.read(max_bytes)
    try:
        sample_text = sample.decode('utf-8-sig')
    except Exception:
        try:
            sample_text = sample.decode('latin1')
        except Exception:
            sample_text = sample.decode('utf-8', errors='ignore')
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text)
        delimiter = dialect.delimiter
    except Exception:
        possible = [',', '\t', ';', '|']
        counts = {d: sample_text.count(d) for d in possible}
        delimiter = max(counts, key=counts.get)
    for enc in ('utf-8-sig', 'utf-8', 'latin1'):
        try:
            df = pd.read_csv(filepath, delimiter=delimiter, encoding=enc, on_bad_lines='skip')
            return df, delimiter
        except Exception:
            continue
    raise Exception("Failed to read CSV with detected delimiter/encodings.")

def normalize_header_if_joined(df):
    if df.shape[1] == 1:
        colname = df.columns[0]
        colname_clean = str(colname).strip()
        if (',' in colname_clean) or ('\t' in colname_clean) or (';' in colname_clean):
            sep = ',' if ',' in colname_clean else ('\t' if '\t' in colname_clean else ';')
            header_cols = [c.strip() for c in colname_clean.split(sep) if c.strip() != '']
            if len(header_cols) > 1:
                new_df = df[df.columns[0]].astype(str).str.split(sep, expand=True)
                new_df.columns = header_cols[:new_df.shape[1]]
                return new_df
    return df

# ---------------------------
# 2) Column guessing & normalization
# ---------------------------
def guess_column(cols, keywords):
    cols_lower = [str(c).lower() for c in cols]
    for kw in keywords:
        for orig, low in zip(cols, cols_lower):
            if kw in low:
                return orig
    return None

def to_float_or_nan(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        s = str(x).strip()
        if s == '' or s.lower() == 'nan':
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def to_int_or_zero(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0
        s = str(x).strip()
        if s == '' or s.lower() == 'nan':
            return 0
        return int(float(s))
    except Exception:
        return 0

def load_book_data(filepath):
    df, detected = detect_delimiter_and_read(filepath)
    print(f"Detected delimiter: '{detected}'")
    df = normalize_header_if_joined(df)
    df.columns = [str(c).strip().strip('\ufeff').strip() for c in df.columns]
    print("Columns:", df.columns.tolist())

    title_col = guess_column(df.columns, ['title', 'book', 'name'])
    author_col = guess_column(df.columns, ['author', 'authors', 'writer'])
    genres_col = guess_column(df.columns, ['genre', 'genres', 'shelves', 'categories'])
    desc_col = guess_column(df.columns, ['description', 'desc', 'summary', 'synopsis', 'about'])
    avg_rating_col = guess_column(df.columns, ['avg', 'average', 'rating'])
    num_ratings_col = guess_column(df.columns, ['num', 'count', 'ratings', 'num_ratings', 'rating_count'])
    lang_col = guess_column(df.columns, ['language', 'lang', 'locale'])

    print("\nGuessed mapping:")
    print(" title ->", title_col)
    print(" author ->", author_col)
    print(" genres ->", genres_col)
    print(" description ->", desc_col)
    print(" avg_rating ->", avg_rating_col)
    print(" num_ratings ->", num_ratings_col)
    print(" language ->", lang_col)

    if title_col is None or author_col is None:
        raise KeyError(f"Could not auto-detect title/author columns. Available: {df.columns.tolist()}")

    df['title'] = df[title_col].astype(str)
    df['author'] = df[author_col].astype(str)
    df['genres'] = df[genres_col].astype(str) if genres_col else ''
    df['description'] = df[desc_col].astype(str) if desc_col else ''
    df['language'] = df[lang_col].astype(str) if lang_col else ''

    df['avg_rating'] = df[avg_rating_col].apply(to_float_or_nan) if avg_rating_col else np.nan
    df['num_ratings'] = df[num_ratings_col].apply(to_int_or_zero) if num_ratings_col else 0

    df.replace({'': np.nan, 'nan': np.nan}, inplace=True)
    init = len(df)
    df.dropna(subset=['title', 'author'], inplace=True)
    print(f"Removed {init - len(df)} rows with missing title/author.")

    books = df.to_dict(orient='records')
    for i, b in enumerate(books):
        b.setdefault('id', i)
        b.setdefault('genres', '')
        b.setdefault('description', '')
        b.setdefault('language', '')
        b.setdefault('avg_rating', np.nan)
        b.setdefault('num_ratings', 0)
    print(f"Loaded {len(books)} books from {filepath}")
    return books

# ---------------------------
# 3) Text cleaning & tokens
# ---------------------------
def clean_text(text):
    if isinstance(text, list):
        text = ' '.join(text)
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def genre_tokens(genre_str):
    if not isinstance(genre_str, str) or not genre_str:
        return set()
    tokens = re.split(r'[\|,;/\-\s]+', genre_str.lower())
    tokens = {t.strip() for t in tokens if t.strip()}
    return tokens

# ---------------------------
# 4) TF-IDF training
# ---------------------------
def train_recommender(processed_books, min_df=0.01, max_df=0.85):
    vectorizer = TfidfVectorizer(stop_words='english', min_df=min_df, max_df=max_df)
    corpus = [b['tags'] for b in processed_books]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("TF-IDF shape:", tfidf_matrix.shape)
    return vectorizer, tfidf_matrix, cosine_sim

# ---------------------------
# 5) Title matching
# ---------------------------
def find_title_index(title, all_books, fuzzy_cutoff=0.72):
    title_norm = title.lower().strip()
    titles = [b['title'] for b in all_books]
    for i, t in enumerate(titles):
        if isinstance(t, str) and t.lower().strip() == title_norm:
            return i
    for i, t in enumerate(titles):
        if isinstance(t, str) and title_norm in t.lower():
            return i
    matches = get_close_matches(title, titles, n=1, cutoff=fuzzy_cutoff)
    if matches:
        return titles.index(matches[0])
    return None

# ---------------------------
# 6) MMR + boosts + explanations
# ---------------------------
def detect_series(title):
    if isinstance(title, str) and '(' in title and ')' in title:
        inside = title.split('(',1)[1].split(')',1)[0]
        if '#' in inside or 'series' in inside.lower():
            return inside.strip()
    return None

def mmr_rerank(candidate_indices, similarity_vector, pairwise_sim_matrix,
               lambda_diversity=0.7, k=10):
    if len(candidate_indices) == 0:
        return []
    selected = []
    candidates = list(candidate_indices)
    sim_scores = {i: float(similarity_vector[i]) for i in candidates}
    pair_sim = np.array(pairwise_sim_matrix)
    while candidates and len(selected) < k:
        mmr_scores = {}
        for c in candidates:
            relevance = sim_scores.get(c, 0.0)
            if not selected:
                diversity_term = 0.0
            else:
                max_sim_to_sel = max(pair_sim[c, s] for s in selected)
                diversity_term = max_sim_to_sel
            mmr_score = lambda_diversity * relevance - (1 - lambda_diversity) * diversity_term
            mmr_scores[c] = mmr_score
        best = max(mmr_scores.items(), key=lambda x: x[1])[0]
        selected.append(best)
        candidates.remove(best)
    return selected

def rerank_with_real_signals(liked_indices, avg_sim_vector, cosine_sim_matrix, all_books,
                             top_n_candidates=300, final_k=10,
                             lambda_diversity=0.72, author_boost=0.12, series_boost=0.15,
                             language_prefer='english'):
    n = len(all_books)
    candidate_idx_scores = sorted([(i, avg_sim_vector[i]) for i in range(n) if i not in liked_indices],
                                  key=lambda x: x[1], reverse=True)[:top_n_candidates]
    candidates = [i for i, s in candidate_idx_scores]
    sim_scores = np.zeros(n)
    for i, s in candidate_idx_scores:
        sim_scores[i] = float(s)

    authors = [str(b.get('author','')).lower() for b in all_books]
    titles = [str(b.get('title','')) for b in all_books]
    series = [detect_series(t) for t in titles]
    languages = [str(b.get('language','')) for b in all_books]

    liked_authors = set(authors[i] for i in liked_indices if authors[i])
    liked_series = set(series[i] for i in liked_indices if series[i])

    boosted_sim = sim_scores.copy()
    for c in candidates:
        boost = 0.0
        if authors[c] in liked_authors and authors[c] != '':
            boost += author_boost
        if series[c] and series[c] in liked_series:
            boost += series_boost
        boosted_sim[c] = boosted_sim[c] + boost

    if language_prefer:
        for c in candidates:
            lang = languages[c]
            if lang and language_prefer.lower() in lang.lower():
                boosted_sim[c] += 0.03

    reranked_idx = mmr_rerank(candidate_indices=candidates,
                              similarity_vector=boosted_sim,
                              pairwise_sim_matrix=cosine_sim_matrix,
                              lambda_diversity=lambda_diversity,
                              k=final_k)

    results = []
    for idx in reranked_idx:
        reasons = []
        if liked_indices:
            contrib_sims = [(li, cosine_sim_matrix[li, idx]) for li in liked_indices]
            best_li, best_sim = max(contrib_sims, key=lambda x: x[1])
            reasons.append(f"Similar to '{all_books[best_li]['title']}' (sim={best_sim:.2f})")
        if authors[idx] in liked_authors:
            reasons.append(f"By same author: {all_books[idx]['author']}")
        if series[idx] and series[idx] in liked_series:
            reasons.append(f"From same series: {series[idx]}")
        liked_genres = set()
        for li in liked_indices:
            liked_genres |= set(str(all_books[li].get('genres','')).lower().split(','))
        candidate_genres = set(str(all_books[idx].get('genres','')).lower().split(','))
        shared = [g.strip() for g in candidate_genres.intersection(liked_genres) if g.strip()]
        if shared:
            reasons.append("Shared genres: " + ", ".join(shared[:3]))
        results.append({
            'id': all_books[idx].get('id'),
            'title': all_books[idx].get('title'),
            'author': all_books[idx].get('author'),
            'genres': all_books[idx].get('genres'),
            'similarity': round(float(sim_scores[idx]), 4),
            'final_score_est': round(float(boosted_sim[idx]), 4),
            'explanation': "; ".join(reasons) if reasons else "Similar content"
        })
    return results

# ---------------------------
# 7) Full pipeline: unzip -> csv -> run
# ---------------------------
if __name__ == "__main__":
    zip_path = r"C:\Users\saanv\Downloads\archive (6).zip"
    extract_path = r"C:\Users\saanv\Downloads\archive_extracted"

    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    if not os.path.exists(extract_path):
        print(f"Extracting {zip_path} -> {extract_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_path)
        print("Extraction complete.")
    else:
        print("Using extracted folder:", extract_path)

    csv_file = None
    for root, _, files in os.walk(extract_path):
        for f in files:
            if f.lower().endswith('.csv'):
                csv_file = os.path.join(root, f)
                break
        if csv_file:
            break
    if csv_file is None:
        raise FileNotFoundError(f"No CSV found in {extract_path}")
    print("Found CSV:", csv_file)

    # Load / normalize
    books_raw = load_book_data(csv_file)

    # Preprocess: build tags and ensure numeric fields normalized upstream
    processed = []
    for b in books_raw:
        tags = " ".join([
            clean_text(b.get('title','')),
            clean_text(b.get('author','')),
            clean_text(b.get('genres','')),
            clean_text(b.get('description',''))
        ])
        processed.append({
            'id': b.get('id'),
            'title': b.get('title'),
            'author': b.get('author'),
            'genres': b.get('genres'),
            'description': b.get('description'),
            'avg_rating': b.get('avg_rating'),
            'num_ratings': b.get('num_ratings'),
            'language': b.get('language',''),
            'tags': tags
        })
    print("Preprocessing done. Items:", len(processed))

    # Train TF-IDF and cosine similarity
    vectorizer, tfidf_matrix, cosine_sim = train_recommender(processed)
    # Ensure cosine_sim is numpy array
    cosine_sim = np.array(cosine_sim)

    # Show sample titles
    print("\nSample titles (first 20):")
    for i, b in enumerate(processed[:20], 1):
        print(f"{i}. {b['title']}")

    # DEMO: use liked titles and produce reranked recommendations
    liked_titles = ["notebook"]  # change as you like
    liked_indices = []
    for t in liked_titles:
        idx = find_title_index(t, processed, fuzzy_cutoff=0.72)
        if idx is None:
            print(f"Warning: liked title '{t}' not found.")
        else:
            print(f"Matched '{t}' => {processed[idx]['title']}")
            liked_indices.append(idx)

    if not liked_indices:
        print("No liked titles matched — update liked_titles to titles from sample above.")
    else:
        # compute avg similarity vector
        n = len(processed)
        avg_sim = np.zeros(n)
        for idx in liked_indices:
            avg_sim += cosine_sim[idx]
        avg_sim /= len(liked_indices)

        # rerank with real signals
        recs = rerank_with_real_signals(liked_indices=liked_indices,
                                        avg_sim_vector=avg_sim,
                                        cosine_sim_matrix=cosine_sim,
                                        all_books=processed,
                                        top_n_candidates=300,
                                        final_k=8,
                                        lambda_diversity=0.72,
                                        author_boost=0.12,
                                        series_boost=0.15,
                                        language_prefer='english')

        print(f"\n--- Final recommendations (MMR + boosts) for liked: {liked_titles} ---")
        for r in recs:
            print(f"- {r['title']} by {r['author']} | genres: {r['genres']} | sim: {r['similarity']} | boosted: {r['final_score_est']} | reason: {r['explanation']}")
