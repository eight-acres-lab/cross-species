"""Point-estimate source selection from historical out-of-fold prediction gains.

No new source response or target human response is accepted at prediction time.
Observed historical gains and availability must be supplied by the caller. This
interface does not establish confidence, safety, or actual species preferences.
"""
import numpy as np


def _features(value):
    value = np.array(value, dtype=float, copy=True)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError("premeasurement_features must be a nonempty (N,F) matrix")
    if not np.isfinite(value).all():
        raise ValueError("premeasurement_features must be finite")
    return value


class SourceUtilitySelector:
    """One ridge per source, fitted only to that source's observed historical gains.

    Each source has its own training feature mean/scale. min_observations is a
    bookkeeping threshold on supplied rows, not a power or independence guarantee.
    Historical gains must come from OOF predictions that simulate the full chosen
    source/calibration pipeline. Split provenance remains the caller's responsibility.
    """

    def __init__(self, source_names=None, alpha=1.0, min_observations=3):
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if (isinstance(min_observations, bool)
                or not isinstance(min_observations, (int, np.integer))
                or min_observations < 2):
            raise ValueError("min_observations must be an integer >= 2")
        self._requested_names = None if source_names is None else list(source_names)
        self.alpha = float(alpha)
        self.min_observations = int(min_observations)

    def fit(self, premeasurement_features, historical_gain, observed_mask):
        features = _features(premeasurement_features)
        gains = np.array(historical_gain, dtype=float, copy=True)
        mask = np.array(observed_mask, copy=True)
        if (gains.ndim != 2 or gains.shape[0] != len(features) or gains.shape[1] < 1
                or mask.shape != gains.shape or mask.dtype.kind != "b"):
            raise ValueError("require historical_gain (N,S) and an equal-shaped boolean observed_mask")
        if not np.isfinite(gains[mask]).all():
            raise ValueError("observed historical gains must be finite")
        names = (["source_%d" % j for j in range(gains.shape[1])]
                 if self._requested_names is None else self._requested_names.copy())
        if (len(names) != gains.shape[1] or any(not isinstance(n, str) or not n for n in names)
                or len(set(names)) != len(names)):
            raise ValueError("source_names must be unique nonempty strings matching the S columns")
        self.source_names_ = names
        self.n_features_in_ = features.shape[1]
        self.observation_counts_ = mask.sum(axis=0)
        self.source_fitted_ = self.observation_counts_ >= self.min_observations
        self.models_ = []
        for j, fitted in enumerate(self.source_fitted_):
            if not fitted:
                self.models_.append(None)
                continue
            x, y = features[mask[:, j]], gains[mask[:, j], j]
            mean, scale = x.mean(axis=0), x.std(axis=0)
            scale[scale < 1e-8] = 1.0
            standardized, intercept = (x - mean) / scale, y.mean()
            coef = np.linalg.lstsq(
                np.vstack([standardized, np.sqrt(self.alpha) * np.eye(self.n_features_in_)]),
                np.r_[y - intercept, np.zeros(self.n_features_in_)], rcond=None)[0]
            self.models_.append((mean.copy(), scale.copy(), coef.copy(), float(intercept)))
        return self

    def predict(self, premeasurement_features):
        """Return (N,S) estimated gains; unfitted sources are NaN, never zero-filled."""
        if not hasattr(self, "models_"):
            raise RuntimeError("fit the selector on historical gains first")
        features = _features(premeasurement_features)
        if features.shape[1] != self.n_features_in_:
            raise ValueError("feature width differs from fitting")
        predictions = np.full((len(features), len(self.source_names_)), np.nan)
        for j, model in enumerate(self.models_):
            if model is not None:
                mean, scale, coef, intercept = model
                predictions[:, j] = ((features - mean) / scale) @ coef + intercept
        return predictions

    def _availability(self, value, n_rows):
        if isinstance(value, (set, frozenset)):
            value = list(value)
        values = np.asarray(value)
        n_sources = len(self.source_names_)
        if values.dtype.kind == "b":
            if values.shape == (n_sources,):
                return np.broadcast_to(values, (n_rows, n_sources)).copy()
            if values.shape == (n_rows, n_sources):
                return values.copy()
        elif values.ndim == 1 and (values.size == 0 or values.dtype.kind in "US"):
            if any(name not in self.source_names_ for name in values.tolist()):
                raise ValueError("available_sources contains an unknown source name")
            mask = np.asarray([name in values.tolist() for name in self.source_names_])
            return np.broadcast_to(mask, (n_rows, n_sources)).copy()
        raise ValueError("available_sources must be an S/NxS boolean mask or a list of known names")

    def select(self, premeasurement_features, available_sources, min_expected_gain=0.0,
               costs=None, cost_weight=0.0):
        """Choose a source or return an explicit point-estimate fallback per row.

        Predicted uncosted gain must strictly exceed min_expected_gain. Choose the
        largest gain - cost_weight*cost among eligible sources, requiring positive
        net utility; ties follow source_names order. Costs must share the caller's
        chosen units. This rule has no per-task no-negative-transfer guarantee.
        """
        if not np.isfinite(min_expected_gain) or not np.isfinite(cost_weight) or cost_weight < 0:
            raise ValueError("threshold must be finite; cost_weight finite and nonnegative")
        predictions = self.predict(premeasurement_features)
        availability = self._availability(available_sources, len(predictions))
        if costs is None:
            cost = np.zeros_like(predictions)
        else:
            cost = np.array(costs, dtype=float, copy=True)
            if cost.shape == (len(self.source_names_),):
                cost = np.broadcast_to(cost, predictions.shape).copy()
            if cost.shape != predictions.shape or not np.isfinite(cost).all() or (cost < 0).any():
                raise ValueError("costs must be finite nonnegative values of shape S or NxS")
        results = []
        for gain, available, row_cost in zip(predictions, availability, cost):
            fitted_available = available & self.source_fitted_
            eligible = fitted_available & (gain > min_expected_gain)
            net = gain - cost_weight * row_cost
            if not available.any():
                reason = "no_available_sources"
            elif not fitted_available.any():
                reason = "no_fitted_available_sources"
            elif not eligible.any():
                reason = "point_estimate_below_gain_threshold"
            elif np.max(net[eligible]) <= 0:
                reason = "nonpositive_cost_adjusted_utility"
            else:
                reason = None
            chosen = None if reason else int(np.argmax(np.where(eligible, net, -np.inf)))
            results.append({
                "chosen_source": None if chosen is None else self.source_names_[chosen],
                "chosen_index": chosen,
                "estimated_gain": None if chosen is None else float(gain[chosen]),
                "net_utility": None if chosen is None else float(net[chosen]),
                "fallback_reason": reason,
                "source_estimates": [
                    {"source": name, "available": bool(available[j]),
                     "fitted": bool(self.source_fitted_[j]),
                     "observations": int(self.observation_counts_[j]),
                     "estimated_gain": float(gain[j]) if self.source_fitted_[j] else None,
                     "cost": float(row_cost[j]),
                     "net_utility": float(net[j]) if self.source_fitted_[j] else None}
                    for j, name in enumerate(self.source_names_)],
            })
        return results
