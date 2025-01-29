import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Amazon reviews dataset
amazon_reviews = [
    "I love this product! It's amazing and works perfectly. Highly recommend it to everyone.",
    "The quality is good, but the delivery was late. Still, I'm happy with my purchase.",
    "Terrible product. It broke within a week. Never buying from this brand again.",
    "The customer service is outstanding. They helped resolve my issue very quickly.",
    "This product is just okay. Nothing special, but not bad either. Fair for the price."
]

# Sentence Tokenization using list comprehension
print("\nStep 4: Sentence Tokenization\n")
all_sentences = [sent.text for review in amazon_reviews for sent in nlp(review).sents]

for review, sentences in zip(amazon_reviews, [nlp(review).sents for review in amazon_reviews]):
    print(f"Review: {review}")
    print(f"Sentences: {[sent.text for sent in sentences]}\n")

# Feature Extraction - TF-IDF using CountVectorizer and TfidfTransformer
print("\nStep 6: Feature Extraction - TF-IDF\n")
count_vectorizer = CountVectorizer()
word_count_matrix = count_vectorizer.fit_transform(amazon_reviews)

tfidf_transformer = TfidfTransformer()
tfidf_matrix = tfidf_transformer.fit_transform(word_count_matrix)

print("Vocabulary (TF-IDF):")
print(count_vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(pd.DataFrame(tfidf_matrix.toarray(), columns=count_vectorizer.get_feature_names_out()))
