import difflib
import nltk


def get_ngram_model(tokens):
    # Initialize the n-gram model
    n = 3
    ngram_model = []

    # Add the n-grams to the model
    ngram_model.extend(nltk.ngrams(tokens, n))

    # Train the n-gram model
    ngram_freq = nltk.FreqDist(ngram_model)

    return ngram_freq


def rank_word(word):
    freq = ngram_freq[(input_seq) + (word)]
    context_freq = ngram_freq[(input_seq, )]
    return freq/(context_freq or 1e-12)


def suggest_word(input_word, tokens):
    # Find words that start with the input word
    candidate_words = [word for word in tokens if word.startswith(input_word)]
    if not candidate_words:
        # Try splitting the input word into n-grams and finding candidate words for each n-gram
        n = 1
        ngrams = [input_word[i:i+n] for i in range(len(input_word)-n+1)]
        for ngram in ngrams:
            candidate_words.extend([word for word in tokens if word.startswith(ngram)])
    
    if not candidate_words:
        return []


    # Sort the candidate words by similarity to the input word
    similarity_scores = [difflib.SequenceMatcher(None, input_word, word).ratio() for word in candidate_words]
    matched_words = [word for _, word in sorted(zip(similarity_scores, candidate_words), reverse=True)]

    return matched_words[:5]


def predict(in_word):
    return suggest_word(in_word, tokens)


# Load text data
with open('corpus_words.txt', 'r') as file:
    text = file.read()

# Tokenize text data
tokens = nltk.word_tokenize(text)

# Build n-gram model
ngram_freq = get_ngram_model(tokens)


"""
# Implement word completion
while True:
    input_seq = input("Enter word: ")
    if input_seq == 'q':
        break

    suggested_words = suggest_word(input_seq, tokens)

    if suggested_words:
        print("Suggested words:")
        print(suggested_words)
    else:
        print("No suggestions found.")

"""