from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def normalize_feature_name(value: object) -> str:
    return (
        str(value)
        .replace("\u00a0", " ")
        .replace("\t", " ")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace(".", "")
    )


class LegacyFeatureNameAdapter(TransformerMixin, BaseEstimator):
    """Map canonical public feature names to names expected by a fitted legacy artifact."""

    def __init__(self, canonical_names: Iterable[str], legacy_names: Iterable[str]):
        self.canonical_names = tuple(canonical_names)
        self.legacy_names = tuple(legacy_names)

    def fit(self, X, y=None):
        if len(self.canonical_names) != len(self.legacy_names):
            raise ValueError("Canonical and legacy feature lists must have the same length.")
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        normalized_columns = {
            normalize_feature_name(column): column for column in frame.columns
        }
        output = pd.DataFrame(index=frame.index)
        for canonical, legacy in zip(self.canonical_names, self.legacy_names):
            key = normalize_feature_name(canonical)
            if key not in normalized_columns:
                raise ValueError(f"Missing canonical input feature: {canonical}")
            output[legacy] = frame[normalized_columns[key]].to_numpy()
        return output

    def get_feature_names_out(self, input_features=None):
        return list(self.legacy_names)
