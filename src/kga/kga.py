"""kga.kga -- the KGA facade: Knowability-Guided Adaptation in one object.

``KGA`` ties together the three K-Bound stages into a small, documented API:

    1. ``.evidence(calib, test)``      -> label-free evidence ``Z``  (kga.evidence)
    2. ``.certify_evidence(...)``      -> frozen ``h(Z)`` and certificate
                                          (kga.certificate)
    3. ``.decide(...)``                -> ADAPT / FREEZE / ABSTAIN  (kga.policy)

with ``.explain()`` returning every intermediate quantity for auditing.

Deployment is label free only on the second path: the benefit estimator must
have been fitted on development data and calibrated on disjoint labelled
conditions before deployment.  :meth:`certify` remains available for labelled
paired-benefit audits and for callers that already hold a point estimate.

Example
-------
>>> import numpy as np
>>> from kga import KGA
>>> rng = np.random.default_rng(0)
>>> kga = KGA(alpha=0.1, method="ebern")
>>> # A labelled paired-benefit audit (not the label-free deployment path).
>>> benefits = rng.normal(0.3, 0.1, size=400)
>>> cert = kga.certify(scores=benefits, benefit_range=2.0)
>>> kga.decide(cert).value in {"ADAPT", "FREEZE", "ABSTAIN"}
True

``benefit_range`` is required for the ``ebern``/``hoeffding`` conventions: it is
the *a priori* support width ``b - a``, and estimating it from the sample voids
the Maurer-Pontil / Hoeffding guarantee (panel findings F1-12, F2-12).  For
paired 0/1 losses it is ``2.0``.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from kga._validation import as_float_array
from kga.benefit import BenefitEstimator
from kga.certificate import (
    Certificate,
    conformal_split,
    empirical_bernstein,
    evalue_anytime,
    hoeffding,
)
from kga.evidence import Evidence, compute_evidence
from kga.policy import Decision, decide

#: Mapping from the public ``method`` name to the batch certificate estimator.
_BATCH_ESTIMATORS = {
    "ebern": empirical_bernstein,
    "hoeffding": hoeffding,
    "evalue": evalue_anytime,
}


def _finite_or_none(value: float) -> float | None:
    """Use JSON null for unavailable numerical fields, not NaN/Infinity."""
    number = float(value)
    return number if np.isfinite(number) else None


class KGA:
    """Knowability-Guided Adaptation gate.

    Parameters
    ----------
    alpha : float, default=0.1
        Requested miscoverage level. False-adapt/false-freeze control is
        conditional on coverage for the certificate's declared target and the
        required sampling, calibration, and transfer assumptions. Measured-cell
        coverage is not automatically population-benefit coverage.
    method : str, default='ebern'
        Default certificate estimator when paired benefit *scores* are supplied
        to :meth:`certify`.  One of ``'ebern'`` (empirical-Bernstein, the default
        batch certificate), ``'hoeffding'``, or ``'evalue'`` (the specified
        bounded-stream e-process with fixed declared nulls and predictable bets,
        not arbitrary repeated deployment decisions).
        When two scalar risks ``adapt_risk``/``freeze_risk`` are supplied with
        calibration residuals, the split-conformal estimator is used instead.

    Attributes
    ----------
    alpha : float
    method : str
    last_evidence : Optional[Evidence]
        The most recent :class:`Evidence` produced by :meth:`evidence`.
    last_certificate : Optional[Certificate]
        The most recent :class:`Certificate` produced by :meth:`certify`.
    last_decision : Optional[Decision]
        The most recent :class:`Decision` produced by :meth:`decide`.
    """

    def __init__(self, alpha: float = 0.1, method: str = "ebern") -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if method not in _BATCH_ESTIMATORS:
            raise ValueError(f"method must be one of {sorted(_BATCH_ESTIMATORS)}, got {method!r}")
        self.alpha = float(alpha)
        self.method = method
        self.last_evidence: Evidence | None = None
        self.last_certificate: Certificate | None = None
        self.last_decision: Decision | None = None
        self.last_estimator_artifact_sha256: str | None = None
        self.last_protocol_sha256: str | None = None
        self.last_evidence_schema_version: str | None = None

    def _invalidate_attempt(self) -> Evidence | None:
        """Clear cached authority before a new operation can fail.

        Callers may restore the evidence for a successful certificate, but a
        failed attempt must never leave an earlier ADAPT/FREEZE reusable via
        ``decide()``. Validation errors remain explicit at this low-level API.
        """
        previous_evidence = self.last_evidence
        self.last_evidence = None
        self.last_certificate = None
        self.last_decision = None
        self.last_estimator_artifact_sha256 = None
        self.last_protocol_sha256 = None
        self.last_evidence_schema_version = None
        return previous_evidence

    def _remember_certificate(self, certificate: Certificate, evidence: Evidence | None = None) -> Certificate:
        self.last_evidence = evidence
        self.last_certificate = certificate
        return certificate

    # ------------------------------------------------------------------
    # Stage 1: label-free evidence
    # ------------------------------------------------------------------
    def evidence(self, calib: np.ndarray, test: np.ndarray, **kwargs: object) -> Evidence:
        """Compute the label-free evidence ``Z`` (delegates to :func:`compute_evidence`).

        Parameters
        ----------
        calib, test : array-like
            Calibration and unlabelled test detector scores; see
            :func:`kga.evidence.compute_evidence`.
        **kwargs
            Forwarded to :func:`compute_evidence` (e.g. ``bins``, ``extra``).

        Returns
        -------
        Evidence
        """
        self._invalidate_attempt()
        ev = compute_evidence(calib, test, **kwargs)  # type: ignore[arg-type]
        self.last_evidence = ev
        return ev

    # ------------------------------------------------------------------
    # Stage 2: finite-sample certificate
    # ------------------------------------------------------------------
    def certify_evidence(
        self,
        estimator: BenefitEstimator,
        *,
        protocol_sha256: str,
        features: dict[str, float] | None = None,
        evidence_schema_version: str | None = None,
    ) -> Certificate:
        """Run the label-free ``Z -> Delta_hat -> interval`` deployment path.

        The estimator is frozen and carries residuals from a disjoint labelled
        calibration split.  At deployment this method reads only the supplied
        feature values; the protocol and evidence-schema identities must match
        the fitted artifact exactly.  Any mismatch raises instead of silently
        reusing a calibration radius.

        Parameters
        ----------
        estimator : BenefitEstimator
            Frozen model and disjoint absolute calibration residuals.
        protocol_sha256 : str
            Digest of the active checkpoint/adapter/split/feature protocol.
        features : mapping, optional
            Protocol-specific label-free feature map.  If omitted, the most
            recent generic :class:`Evidence` from :meth:`evidence` is used.
        evidence_schema_version : str, optional
            Required with a custom feature map; inferred from ``Evidence`` for
            the generic score-evidence path.

        Returns
        -------
        Certificate
            Exact-rank split-conformal certificate around ``estimator.predict``.

        Notes
        -----
        The statistical coverage claim is conditional on the calibration
        residual and deployment unit being exchangeable (or on another stated
        valid transfer argument).  Schema validation does not prove that
        assumption.
        """

        previous_evidence = self._invalidate_attempt()
        if features is None:
            if previous_evidence is None:
                raise ValueError("No evidence available; call evidence(...) or pass features.")
            features = previous_evidence.to_mapping()
            evidence_schema_version = previous_evidence.schema_version
        else:
            # Custom features do not inherit unrelated cached score evidence.
            previous_evidence = None
            if evidence_schema_version is None:
                raise ValueError("evidence_schema_version is required with custom features")
        if not isinstance(estimator, BenefitEstimator):
            raise TypeError("estimator must implement the frozen BenefitEstimator protocol")
        # Enforce the public contract here as well as in the reference linear
        # estimator. A custom predictor must not silently impute missing inputs.
        if protocol_sha256 != estimator.protocol_sha256:
            raise ValueError("protocol SHA-256 does not match the frozen benefit estimator")
        if evidence_schema_version != estimator.evidence_schema_version:
            raise ValueError("evidence schema does not match the frozen benefit estimator")
        names = tuple(estimator.feature_names)
        if not names or len(names) != len(set(names)) or set(features) != set(names):
            raise ValueError("evidence feature mismatch with the frozen estimator")
        feature_values = as_float_array([features[name] for name in names])
        if feature_values.shape != (len(names),) or not np.isfinite(feature_values).all():
            raise ValueError("evidence features must be finite scalar values")
        delta_hat = estimator.predict(
            features,
            evidence_schema_version=str(evidence_schema_version),
            protocol_sha256=protocol_sha256,
        )
        cert = conformal_split(delta_hat, estimator.residuals, alpha=self.alpha)
        # Resolve the final identity before restoring any cached authority: a
        # failing artifact property must also leave the attempt unavailable.
        artifact_sha256 = estimator.artifact_sha256
        self.last_estimator_artifact_sha256 = artifact_sha256
        self.last_protocol_sha256 = protocol_sha256
        self.last_evidence_schema_version = str(evidence_schema_version)
        return self._remember_certificate(cert, previous_evidence)

    def certify(
        self,
        adapt_risk: float | None = None,
        freeze_risk: float | None = None,
        *,
        scores: np.ndarray | None = None,
        calib_residuals: np.ndarray | None = None,
        delta_hat: float | None = None,
        method: str | None = None,
        benefit_range: float | None = None,
    ) -> Certificate:
        """Build a certificate ``Delta_hat +/- epsilon`` for adapting vs freezing.

        Three audit/low-level calling conventions are supported (pick exactly
        one).  None consumes :attr:`last_evidence`; use
        :meth:`certify_evidence` for the label-free deployment path.

        1. **Paired benefits** -- pass ``scores`` = per-sample benefits
           ``X_i = loss(f0_i) - loss(fa_i)`` (positive means adapt helps).  A
           batch estimator (``method`` or ``self.method``) produces the
           certificate using the selected batch bound or bounded-stream
           e-process method.
        2. **Two scalar risks + calibration residuals** -- pass ``adapt_risk``
           and ``freeze_risk`` (so ``delta_hat = freeze_risk - adapt_risk``) and
           ``calib_residuals`` = held-out ``|Delta_hat_i - Delta_i|`` errors.
           The cross-task split-conformal radius is used.
        3. **Explicit point estimate + calibration residuals** -- pass
           ``delta_hat`` directly together with ``calib_residuals`` for the
           split-conformal radius.

        Parameters
        ----------
        adapt_risk, freeze_risk : float, optional
            Estimated risks of the adapted and frozen predictors.  Convention 2.
            ``delta_hat = R(f0) - R(fa) = freeze_risk - adapt_risk``.
        scores : array-like, optional
            Per-sample paired benefits.  Convention 1.
        calib_residuals : array-like, optional
            Calibration residuals for the split-conformal radius (conventions 2
            and 3).
        delta_hat : float, optional
            Explicit benefit point estimate.  Convention 3.
        method : str, optional
            Override the batch estimator for convention 1.
        benefit_range : float, optional
            *A priori* range ``R = b - a`` of the paired benefits.  **Required**
            for ``ebern``/``hoeffding`` (convention 1) -- there is no safe
            data-estimated default, because substituting the observed
            ``max - min`` makes the deviation term data-dependent and voids the
            finite-sample guarantee (panel findings F1-12 / F2-12).  Pass ``2.0``
            for ``|p - y|`` paired 0/1 losses.  Ignored by ``evalue`` and by the
            conformal conventions.

        Returns
        -------
        Certificate

        Raises
        ------
        ValueError
            If the supplied arguments do not match exactly one convention, or if
            ``benefit_range`` is missing for ``ebern``/``hoeffding``.
        """
        previous_evidence = self._invalidate_attempt()
        # Convention 1 (paired benefits) takes precedence when `scores` is given.
        if scores is not None:
            if method is None:
                method = self.method
            if method not in _BATCH_ESTIMATORS:
                raise ValueError(f"method must be one of {sorted(_BATCH_ESTIMATORS)}, got {method!r}")
            estimator = _BATCH_ESTIMATORS[method]
            if method in ("ebern", "hoeffding"):
                cert = estimator(  # type: ignore[operator]
                    scores, alpha=self.alpha, benefit_range=benefit_range
                )
            else:  # evalue
                cert = estimator(scores, alpha=self.alpha)  # type: ignore[operator]
            return self._remember_certificate(cast(Certificate, cert), previous_evidence)

        # Conformal conventions (2 or 3): need a point estimate + residuals.
        if delta_hat is None:
            if adapt_risk is None or freeze_risk is None:
                raise ValueError(
                    "Provide either `scores` (paired benefits), or `delta_hat` + "
                    "`calib_residuals`, or both `adapt_risk` and `freeze_risk` + "
                    "`calib_residuals`."
                )
            delta_hat = float(freeze_risk) - float(adapt_risk)
        if calib_residuals is None:
            raise ValueError(
                "`calib_residuals` is required for the conformal certificate "
                "(pass held-out |Delta_hat - Delta| residuals)."
            )
        cert = conformal_split(float(delta_hat), calib_residuals, alpha=self.alpha)
        return self._remember_certificate(cert, previous_evidence)

    def certify_probe(
        self,
        benefits: np.ndarray,
        *,
        k: int | None = None,
        seed: int = 0,
        method: str | None = None,
        benefit_range: float | None = None,
    ) -> Certificate:
        """Target-label-light certificate from a labeled micro-probe.

        Uses per-sample paired benefits (positive means adapt/fuse helps) on a
        held-out probe set.  When ``k`` is smaller than the supplied pool,
        ``k`` benefits are drawn without replacement (deterministic given
        ``seed``) to simulate a fixed-size deployment probe.

        Parameters
        ----------
        benefits : array-like
            Per-sample benefits ``X_i = loss(f0_i) - loss(fa_i)`` or
            placement-benefit differences for AUROC routing.
        k : int, optional
            Positive integer probe size. ``None`` or ``k >= len(benefits)``
            uses the full pool. Zero, negative, and non-integer budgets are
            rejected rather than silently using more probe data.
        seed : int
            RNG seed for subsampling when ``k < len(benefits)``.
        method, benefit_range
            Forwarded to the batch estimator (default ``self.method``).
            ``benefit_range`` is required for ``ebern``/``hoeffding``; see
            :meth:`certify`.

        Returns
        -------
        Certificate
        """
        previous_evidence = self._invalidate_attempt()
        if k is not None and (not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1):
            raise ValueError("k must be a positive integer or None")
        pool = as_float_array(benefits).ravel()
        if pool.size == 0:
            raise ValueError("benefits must be non-empty")
        if not np.all(np.isfinite(pool)):
            raise ValueError("benefits must be finite")
        if k is not None and k < pool.size:
            rng = np.random.default_rng(seed)
            idx = rng.choice(pool.size, size=k, replace=False)
            pool = pool[idx]
        if method is None:
            method = self.method
        if method not in _BATCH_ESTIMATORS:
            raise ValueError(f"method must be one of {sorted(_BATCH_ESTIMATORS)}, got {method!r}")
        estimator = _BATCH_ESTIMATORS[method]
        if method in ("ebern", "hoeffding"):
            cert = estimator(pool, alpha=self.alpha, benefit_range=benefit_range)  # type: ignore[operator]
        else:
            cert = estimator(pool, alpha=self.alpha)  # type: ignore[operator]
        return self._remember_certificate(cast(Certificate, cert), previous_evidence)

    # ------------------------------------------------------------------
    # Stage 3: trichotomy decision
    # ------------------------------------------------------------------
    def decide(self, certificate: Certificate | None = None) -> Decision:
        """Apply the trichotomy to a certificate (default: the last one built).

        Parameters
        ----------
        certificate : Certificate, optional
            The certificate to decide on.  If ``None``, uses
            :attr:`last_certificate` (and raises if there is none).

        Returns
        -------
        Decision
            ADAPT / FREEZE / ABSTAIN. False-adapt control is inherited only when
            the certificate's stated coverage assumptions hold; the threshold
            function itself cannot verify them.
        """
        previous_certificate = self.last_certificate
        previous_identity = (
            self.last_estimator_artifact_sha256,
            self.last_protocol_sha256,
            self.last_evidence_schema_version,
        )
        previous_evidence = self._invalidate_attempt()
        if certificate is None:
            certificate = previous_certificate
        if certificate is None:
            raise ValueError("No certificate available; call certify(...) first or pass one.")
        dec = decide(certificate, alpha=self.alpha)
        self.last_certificate = certificate
        if certificate is previous_certificate:
            self.last_evidence = previous_evidence
            (
                self.last_estimator_artifact_sha256,
                self.last_protocol_sha256,
                self.last_evidence_schema_version,
            ) = previous_identity
        self.last_decision = dec
        return dec

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def explain(self) -> dict:
        """Return all intermediate quantities of the last gate evaluation.

        Returns
        -------
        dict
            Keys: ``alpha``, ``method``, ``evidence`` (or ``None``),
            ``certificate`` (or ``None``) with ``delta_hat``/``epsilon``/
            ``lower``/``upper``/``method``/``n``, and ``decision`` (or ``None``).
            All values are strict-JSON-serialisable. Nonfinite numerical fields
            (such as an unavailable infinite radius) are represented by ``None``
            / JSON null. The original certificate retains its infinite radius
            for the ABSTAIN rule; producing this report does not change it.
        """
        ev = self.last_evidence
        cert = self.last_certificate
        dec = self.last_decision
        evidence_dict = None
        if ev is not None:
            evidence_dict = {
                "ks_mean": _finite_or_none(ev.ks_mean),
                "ks_max": _finite_or_none(ev.ks_max),
                "disagree": _finite_or_none(ev.disagree),
                "entropy_shift": _finite_or_none(ev.entropy_shift),
                "conf_shift": _finite_or_none(ev.conf_shift),
                "calib_entropy": _finite_or_none(ev.calib_entropy),
                "test_entropy": _finite_or_none(ev.test_entropy),
                "calib_conf": _finite_or_none(ev.calib_conf),
                "test_conf": _finite_or_none(ev.test_conf),
                "ess": _finite_or_none(ev.ess),
                "ess_frac": _finite_or_none(ev.ess_frac),
                "n_calib": int(ev.n_calib),
                "n_test": int(ev.n_test),
                "n_detectors": int(ev.n_detectors),
            }
        certificate_dict = None
        if cert is not None:
            certificate_dict = {
                "delta_hat": _finite_or_none(cert.delta_hat),
                "epsilon": _finite_or_none(cert.epsilon),
                "lower": _finite_or_none(cert.lower),
                "upper": _finite_or_none(cert.upper),
                "method": cert.method,
                "alpha": _finite_or_none(cert.alpha),
                "n": int(cert.n),
            }
        return {
            "alpha": _finite_or_none(self.alpha),
            "method": self.method,
            "evidence": evidence_dict,
            "certificate": certificate_dict,
            "decision": dec.value if dec is not None else None,
            "estimator_artifact_sha256": self.last_estimator_artifact_sha256,
            "protocol_sha256": self.last_protocol_sha256,
            "evidence_schema_version": self.last_evidence_schema_version,
        }
