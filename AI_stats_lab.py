"""
AIstats_lab.py

Student starter file for:
1. Naive Bayes spam classification
2. K-Nearest Neighbors on Iris
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    return float(np.mean(y_true == y_pred))


# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():

    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    # 1. Tokenize
    tokenized = []
    for text in texts:
        tokenized.append(text.split())

    # 2. Vocabulary
    vocab = []
    for words in tokenized:
        for w in words:
            if w not in vocab:
                vocab.append(w)

    # 3. Priors
    priors = {}
    total_docs = len(labels)

    count_1 = 0
    count_0 = 0
    for label in labels:
        if label == 1:
            count_1 += 1
        else:
            count_0 += 1

    priors[1] = count_1 / total_docs
    priors[0] = count_0 / total_docs

    # 4. Word counts (NO defaultdict)
    word_counts = {0: {}, 1: {}}
    total_words = {0: 0, 1: 0}

    for i in range(len(tokenized)):
        words = tokenized[i]
        label = labels[i]

        for w in words:
            # manual dictionary update
            if w in word_counts[label]:
                word_counts[label][w] += 1
            else:
                word_counts[label][w] = 1

            total_words[label] += 1

    # 5. Word probabilities
    word_probs = {0: {}, 1: {}}

    for c in [0, 1]:
        for w in vocab:
            if w in word_counts[c]:
                word_probs[c][w] = word_counts[c][w] / total_words[c]
            else:
                word_probs[c][w] = 0

    # 6. Prediction
    test_words = test_email.split()

    scores = {}

    for c in [0, 1]:
        score = priors[c]

        for w in test_words:
            if w in word_probs[c] and word_probs[c][w] > 0:
                score *= word_probs[c][w]
            else:
                score *= 0   # no smoothing

        scores[c] = score

    prediction = 1 if scores[1] > scores[0] else 0

    return priors, word_probs, prediction

# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):

    # 1. Load data
    data = load_iris()
    X = data.data
    y = data.target

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # 3. Euclidean distance
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # 4. Predict function
    def predict(X_train, y_train, X_test, k):
        predictions = []

        for test_point in X_test:
            distances = []

            for i in range(len(X_train)):
                dist = euclidean_distance(test_point, X_train[i])
                distances.append((dist, y_train[i]))

            # sort by distance
            distances.sort(key=lambda x: x[0])

            # get k nearest
            neighbors = distances[:k]

            # majority vote
            labels = [label for _, label in neighbors]
            pred = max(set(labels), key=labels.count)

            predictions.append(pred)

        return np.array(predictions)

    # 5. Predictions
    train_preds = predict(X_train, y_train, X_train, k)
    test_preds = predict(X_train, y_train, X_test, k)

    # 6. Accuracy
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    return train_accuracy, test_accuracy, test_preds