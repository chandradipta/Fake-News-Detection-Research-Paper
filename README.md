# News Credibility Classifier

A machine learning pipeline that automatically classifies news articles as
credible or fabricated using natural language processing techniques.

---

## Project Structure

```
FakeNewsDetectionProject/
├── source_code/
│   └── news_credibility_classifier.py   # Main pipeline script
├── dataset/
│   └── dataset_link.txt                 # Link to the Kaggle dataset
├── diagrams/
│   └── pipeline_flowchart.txt           # Text description of the pipeline
├── papers_referred/
│   └── research_links.txt               # Academic references used
├── presentation/
│   └── project_presentation.pptx        # Slide deck
├── report/
│   └── research_paper.docx              # Full written report
├── results/
│   └── accuracy_results.txt             # Recorded evaluation metrics
└── requirements.txt                     # Python dependencies
```

---

## How It Works

The pipeline processes each article through five sequential stages:

1. **Input construction** — The author name and article title are concatenated
   into a single text field. Missing values are imputed with empty strings.

2. **Text normalisation** — Non-alphabetic characters are stripped, tokens are
   lowercased, English stopwords are removed, and Porter stemming is applied to
   reduce words to their root forms.

3. **TF-IDF vectorisation** — The normalised text is converted into a sparse
   numeric feature matrix using Term Frequency-Inverse Document Frequency
   weighting. Sublinear TF scaling is applied to limit the influence of
   high-frequency terms.

4. **Model training** — Three classifiers are trained on an 80/20
   stratified split:
   - Logistic Regression (L2-regularised, liblinear solver)
   - Multinomial Naive Bayes (Laplace smoothing α = 0.1)
   - Random Forest (100 trees, Gini criterion)

5. **Evaluation** — Each model is evaluated on the held-out test set.
   Accuracy, precision, recall, F1-score, and a confusion matrix are reported.

---

## Dataset

The project uses the **Kaggle Fake News Dataset** (~20,800 labelled articles).

Download it from the link in `dataset/dataset_link.txt` and place the CSV file
in the project root directory before running the script.

Expected filename: `fake_news.csv`

Required columns: `id`, `title`, `author`, `text`, `label`
- `label = 0` → real / credible article
- `label = 1` → fabricated / fake article

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.8 or above is recommended.

---

## Running the Pipeline

Basic usage (defaults to `fake_news.csv` in the current directory):

```bash
python source_code/news_credibility_classifier.py
```

Custom dataset path:

```bash
python source_code/news_credibility_classifier.py --data path/to/dataset.csv
```

Save confusion matrix images for all models:

```bash
python source_code/news_credibility_classifier.py --save-cm
```

All available options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `fake_news.csv` | Path to the labelled CSV file |
| `--test-size` | `0.2` | Fraction of data held out for evaluation |
| `--seed` | `42` | Random seed (for reproducibility) |
| `--save-cm` | off | Save confusion matrix PNGs to disk |

---

## Results

| Model | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| Logistic Regression | 0.954 | 0.951 | 0.952 | **95.2%** |
| Random Forest | 0.938 | 0.931 | 0.934 | 93.5% |
| Naive Bayes | 0.923 | 0.910 | 0.916 | 91.7% |

Logistic Regression achieves the best performance under this feature regime,
which is consistent with findings in the NLP literature for high-dimensional
sparse text representations.

---

## Dependencies

See `requirements.txt`. Core libraries:

- **pandas / numpy** — data handling
- **scikit-learn** — vectorisation, models, and evaluation metrics
- **nltk** — stopword corpus and Porter stemmer
- **matplotlib** — confusion matrix visualisation

---

## References

See `papers_referred/research_links.txt` for the academic papers that informed
the design of this project.

---

## License

This project is released for academic and educational use.
