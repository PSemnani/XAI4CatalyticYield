from os import remove
import numpy as np
import scipy as sp

elements = ['Ba', 'Ca', 'Ce', 'Co', 'Cs', 'Cu', 'Eu', 'Fe', 'Hf', 'K',
            'La', 'Li', 'Mg', 'Mn', 'Mo', 'Na', 'Nd', 'Ni', 'Pd', 'Sr', 'Tb', 'Ti',
            'V', 'W', 'Y', 'Zn', 'Zr']
supports = ['Al2O3', 'BaO', 'CaO', 'CeO2', 'La2O3',
            'MgO', 'SiO2', 'TiO2', 'ZrO2']


def generate_catalysts_from_relevance_scores(relevance_scores, feature_names,
                                           num_candidates,
                                           elem_importance_factor=1,
                                           supp_importance_factor=1):
    """
    Generate catalys candidates from given feature relevance scores.

    Args:
        relevance_scores: a list of feature relevance scores
        feature_names: a list of feature names
        num_candidates: the number of candidates to create
        importance_factor: scaling factor for importance scores, controls variability of candidates
     Returns:
        all_candidates: a list of candidates
    """
    relevance_scores = relevance_scores
    all_candidates = []
    elem_scores = relevance_scores[:len(elements)] * elem_importance_factor
    elem_names = feature_names[:len(elements)]
    supp_scores = relevance_scores[len(elements):] * supp_importance_factor
    supp_names = feature_names[len(elements):]

    # calculate element and support probabilities using a softmax based on the relevance scores
    supp_probs = sp.special.softmax(supp_scores)

    for i in range(num_candidates):
        # separate the element and support names ans scores from the overall names and scores

        supp_idx = np.random.choice(np.arange(len(supp_probs)), p=supp_probs)

        # select up to 3 elements, each time giving the option to choose none, and ensuring no repetition
        ns_elem_idx = np.arange(len(elem_scores))
        selected_elems = []
        for i in range(3):
            if np.random.rand() < (1 / len(ns_elem_idx)):
                selected_elems.append('None')
            else:
                # calculate element probabilities using relevance scores
                elem_probs = sp.special.softmax(elem_scores[ns_elem_idx])
                # select one element and append it
                select_idx = np.random.choice(ns_elem_idx, p=elem_probs)
                selected_elems.append(elem_names[select_idx])
                # remove selected element from list possible indices
                ns_elem_idx = np.setdiff1d(ns_elem_idx, select_idx)

        candidate = selected_elems + [supp_names[supp_idx]]
        all_candidates.append(candidate)

    return all_candidates


def catalyst_string_to_numpy(catalyst_strings, feature_names, remove_duplicates=True):
    """
    Convert catalyst strings to feature vectors usable by ML models.

    Args:
        catalyst_strings: a list of catalyst strings
        feature_names: a list of feature names
        remove_duplicates: whether to remove duplicate features
    Returns:
        catalyst_features: a np array containing the feature vectors (n_samples x n_features)
    """
    catalyst_features = []
    for cat in catalyst_strings:
        feats = np.zeros(len(feature_names))
        for i, elem in enumerate(feature_names):
            if elem in cat:
                feats[i] = 1
        catalyst_features.append(feats)
    catalyst_features = np.array(catalyst_features)
    if remove_duplicates:
        catalyst_features = np.unique(catalyst_features, axis=0)
    return catalyst_features


def numpy_to_catalyst_string(catalyst_features, feature_names):
    """
    Convert catalyst feature vectors to a human readable catalyst string.

    Args:
        catalyst_features: a numpy array containing the feature vectors
        feature_names: a list of feature names
    Returns:
        catalyst_strings: a np array containing the feature vectors (n_samples x n_features)
    """
    catalyst_strings = []
    for cat in catalyst_features:
        catalyst_strings.append(list(feature_names[cat.astype(bool)]))
    return catalyst_strings
