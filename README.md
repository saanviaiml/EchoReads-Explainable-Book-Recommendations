
#  EchoReads — Explainable Book Recommendations

**EchoReads** is a **content-based book recommender** that combines **TF–IDF similarity** with **MMR re-ranking**, **author/series boosts**, **genre overlap weighting**, and **popularity signals** to deliver personalized book suggestions — with short explanations for *why* each book was recommended.

---

##  Features

* **TF–IDF + Cosine Similarity** for base relevance
* **MMR Re-ranking** to improve diversity & avoid duplicates
* **Author & Series Boosts** to prioritize similar works
* **Genre Overlap Weighting** to favor thematic matches
* **Popularity Signal** using `(avg rating × log(num_ratings))`
* **Robust CSV/ZIP Import** with delimiter & encoding detection
* **Fuzzy Title Matching** for flexible user input
* **Language Preference Filter**
* **Human-readable Explanations** for each recommendation

---

##  Project Structure

```
 echo_reads/
 ┣  book_recommender_mmr.py    # Main Python script
 ┣  README.md                  # Project documentation
 ┣  requirements.txt           # Dependencies
 ┗  data/
     ┗ books.csv (or archive.zip) # Your dataset
```

---

##  Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/echo-reads.git
cd echo-reads
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

##  Dataset

* You can use any dataset containing at least:

  * **Title**
  * **Author**
  * *(Optional but recommended)* Genres, Description, Average Rating, Number of Ratings, Language
* Place the CSV or ZIP file in the `data/` folder.
* The script will automatically detect delimiter, encoding, and normalize columns.

---

##  Usage

**Run with Python**

```bash
python book_recommender_mmr.py
```

**Or in Jupyter Notebook**

```python
!pip install pandas numpy scikit-learn
%run book_recommender_mmr.py
```

---

##  Example Output

For liked title `"Notebook"`:

```
--- Final recommendations (MMR + boosts) ---
- The Wedding by Nicholas Sparks | genres: Romance | sim: 0.412 | boosted: 0.532 | reason: By same author; Similar to 'The Notebook'
- Safe Haven by Nicholas Sparks | genres: Romance, Drama | sim: 0.389 | boosted: 0.501 | reason: By same author; Shared genres: romance
```

---

##  Requirements

```
pandas
numpy
scikit-learn
```

*(Python 3.8+ recommended)*

## Future Improvements

* Integrate collaborative filtering
* Add user profile persistence
* Deploy as a web app using **Flask** or **Streamlit**
