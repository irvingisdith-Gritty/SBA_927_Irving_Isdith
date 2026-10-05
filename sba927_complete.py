import nltk
import string
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ============================================================
# SBA927 - TEXT DATA PROCESSING AND NLP ANALYSIS
# ============================================================

print("=" * 60)
print("SBA927 - TEXT DATA PROCESSING AND NLP ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# NLTK resources
# ------------------------------------------------------------
resources = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/words", "words"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
    ("chunkers/maxent_ne_chunker_tab", "maxent_ne_chunker_tab"),
    ("sentiment/vader_lexicon", "vader_lexicon"),
]

for resource_path, download_name in resources:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        print(f"Downloading NLTK resource: {download_name}")
        nltk.download(download_name, quiet=True)

# ------------------------------------------------------------
# Locate the dataset
# ------------------------------------------------------------
possible_files = [
    Path("SBA927.txt"),
    Path("sba927.txt"),
    Path(__file__).with_name("SBA927.txt"),
    Path(__file__).with_name("sba927.txt"),
]

file_path = next((p for p in possible_files if p.exists()), None)

if file_path is None:
    print("\nERROR: SBA927.txt was not found.")
    print("Place SBA927.txt in the same folder as sba927.py.")
    input("\nPress Enter to exit...")
    raise SystemExit

print(f"\nDataset loaded from: {file_path.resolve()}")

with open(file_path, "r", encoding="utf-8") as file:
    dataset = [line.strip() for line in file if line.strip()]

print(f"Number of text records: {len(dataset)}")

# ============================================================
# REQUIREMENT 1: TOKENIZATION AND PREPROCESSING
# ============================================================

def tokenize_and_preprocess(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))

    filtered_tokens = [
        word.lower()
        for word in tokens
        if word.isalpha()
        and word.lower() not in stop_words
        and word not in string.punctuation
    ]

    return filtered_tokens


processed_dataset = [
    tokenize_and_preprocess(text)
    for text in dataset
]

print("\n" + "=" * 60)
print("REQUIREMENT 1: TOKENIZATION AND PREPROCESSING")
print("=" * 60)

for i, tokens in enumerate(processed_dataset):
    print(f"\nOriginal Text {i + 1}:")
    print(dataset[i])
    print(f"Processed Tokens {i + 1}:")
    print(tokens)

# ============================================================
# REQUIREMENT 2: STEMMING AND LEMMATIZATION
# ============================================================

def stem_and_lemmatize(tokens):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return stemmed_tokens, lemmatized_tokens


stemmed_and_lemmatized_dataset = [
    stem_and_lemmatize(tokens)
    for tokens in processed_dataset
]

print("\n" + "=" * 60)
print("REQUIREMENT 2: STEMMING AND LEMMATIZATION")
print("=" * 60)

for i, (stemmed_tokens, lemmatized_tokens) in enumerate(
    stemmed_and_lemmatized_dataset
):
    print(f"\nRecord {i + 1}")
    print(f"Original Tokens: {processed_dataset[i]}")
    print(f"Stemmed Tokens: {stemmed_tokens}")
    print(f"Lemmatized Tokens: {lemmatized_tokens}")

# ============================================================
# REQUIREMENT 3: PART-OF-SPEECH (POS) TAGGING
# ============================================================

print("\n" + "=" * 60)
print("REQUIREMENT 3: PART-OF-SPEECH TAGGING")
print("=" * 60)

pos_results = []

for i, text in enumerate(dataset):
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    pos_results.append(tagged_tokens)

    print(f"\nRecord {i + 1}:")
    print(tagged_tokens)

# ============================================================
# REQUIREMENT 4: NAMED ENTITY RECOGNITION (NER)
# ============================================================

print("\n" + "=" * 60)
print("REQUIREMENT 4: NAMED ENTITY RECOGNITION (NER)")
print("=" * 60)

for i, text in enumerate(dataset):
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    ner_result = nltk.ne_chunk(tagged_tokens)

    print(f"\nOriginal Text {i + 1}:")
    print(text)
    print("\nNamed Entities:")

    found_entity = False

    for entity in ner_result:
        if isinstance(entity, nltk.Tree):
            entity_name = " ".join(
                word for word, tag in entity.leaves()
            )
            entity_type = entity.label()
            print(f"  {entity_name} -> {entity_type}")
            found_entity = True

    if not found_entity:
        print("  No named entities found.")

# ============================================================
# REQUIREMENT 5: SENTIMENT ANALYSIS
# ============================================================

from nltk.sentiment import SentimentIntensityAnalyzer

print("\n" + "=" * 60)
print("REQUIREMENT 5: SENTIMENT ANALYSIS")
print("=" * 60)

sia = SentimentIntensityAnalyzer()

sentiment_results = []

for i, text in enumerate(dataset):
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    sentiment_results.append((sentiment, scores))

    print(f"\nRecord {i + 1}")
    print(f"Sentiment: {sentiment}")
    print(f"Scores: {scores}")

# ============================================================
# REQUIREMENT 6: PRE-TRAINED MODEL
# ============================================================
#
# This section uses a Hugging Face pre-trained sentiment model.
# Install the required packages in PowerShell if necessary:
#
#   python -m pip install transformers torch
#
# The program will continue normally if those packages are not
# installed, and the NLTK sentiment results above will still run.
# ============================================================

print("\n" + "=" * 60)
print("REQUIREMENT 6: PRE-TRAINED MODEL")
print("=" * 60)

try:
    from transformers import pipeline

    print("Loading pre-trained DistilBERT sentiment model...")
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    for i, text in enumerate(dataset):
        result = classifier(text[:512])[0]

        print(f"\nRecord {i + 1}")
        print(f"Model Label: {result['label']}")
        print(f"Confidence: {result['score']:.4f}")

except ImportError:
    print("\nThe Transformers library is not installed.")
    print("To enable the pre-trained model, run:")
    print("python -m pip install transformers torch")
    print("\nThe rest of the assignment completed successfully.")

except Exception as error:
    print("\nThe pre-trained model could not be loaded.")
    print(f"Reason: {error}")
    print("\nThe rest of the assignment completed successfully.")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"Dataset records processed: {len(dataset)}")
print("Completed: preprocessing")
print("Completed: stemming and lemmatization")
print("Completed: POS tagging")
print("Completed: named entity recognition")
print("Completed: sentiment analysis")

print("\nIf the Transformers package is installed, the")
print("pre-trained DistilBERT analysis is also completed.")

print("\nDone!")
