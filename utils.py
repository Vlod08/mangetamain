def clean_and_tokenize(text):
    import re
    """
    Cleaning a document with:
        - Lowercase        
        - Removing numbers with regular expressions
        - Removing punctuation with regular expressions
        - Removing other artifacts
    And separate the document into words by simply splitting at spaces
    Params:
        text (string): a sentence or a document
    Returns:
        tokens (list of strings): the list of tokens (word units) forming the document
    """
    # Lowercase
    try:
        text = text.lower()
        # Remove numbers
        text = re.sub(r"[0-9]+", "", text)
        # Remove punctuation
        REMOVE_PUNCT = re.compile("[.;:!\'?,\"()\[\]]")
        text = REMOVE_PUNCT.sub("", text)
        # Remove words beginning by @ ? (Good choice ?)
        # text = re.sub(r'()@\w+', r'\1', text)
        tokens = text.split()        
    except Exception as e:
        print(f"Error in text processing: {e} with text: {text}")
    return tokens

def count_words(texts, voc = None):
    import numpy as np
    """Vectorize text : return count of each word in the text snippets

    Parameters
    ----------
    texts : list of str
        The texts
    Returns
    -------
    vocabulary : dict
        A dictionary that points to an index in counts for each word.
    counts : ndarray, shape (n_samples, n_features)
        The counts of each word in each text.
    """
    n_samples = len(texts)
    
    # If the vocabulary is not known, we need to build it
    if voc == None:
        words =set()
        for i,t in enumerate(texts):
            if i%1000==0:
                print(f'Vocab Update : Processing document {i}/{n_samples}')
            words = words.union(set(clean_and_tokenize(t)))
        n_features = len(words)
        
        vocabulary = dict(zip(words, range(n_features)))
        
    # If it's given, it's quite easier
    else:
        vocabulary = voc
        n_features = len(voc)
    
    # Creating the matrix counts
    counts = np.zeros((n_samples, n_features))
    
    # Filling the matrix by iterating over the documents and counting the words
    for k, t in enumerate(texts): 
        if k%1000==0:
            print(f'Count Vectorization : Processing document {k}/{n_samples}')
        for w in clean_and_tokenize(t):
            counts[k][vocabulary[w]] += 1.
    
    return vocabulary, counts

def tfidf_transform(bow):
    import numpy as np

    """
    Apply inverse document frequencies to a bag-of-words representation.

    Parameters
    ----------
    bow : ndarray or sparse matrix, shape (n_docs, n_words)
        Bag-of-words matrix with word counts.

    Returns
    -------
    tf_idf : ndarray
        TF-IDF weighted representation of same shape as bow.
    """
    # --- IDF ---
    d = bow.shape[0]                         # number of documents
    in_doc = (bow > 0).sum(axis=0)           # number of docs each term appears in
    idf = np.log((d) / (1 + in_doc)) + 1     # smoothed IDF (+1 avoids div by zero)

    # --- TF ---
    sum_vec = bow.sum(axis=1)                # total word count per doc
    tf = bow / np.maximum(sum_vec[:, None], 1)  # normalize per document

    # --- TF-IDF ---
    tf_idf = tf * idf                        # elementwise multiply
    return tf_idf


