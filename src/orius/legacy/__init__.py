"""Legacy adapters preserved for extensibility demonstrations only.

The modules in this package (``aerospace_adapter``, ``industrial_adapter``,
``navigation_adapter``) are **not part of the defended ORIUS evidence
package**.  The IEEE manuscript defends exactly three rows: Battery (witness),
AV (bounded), and Healthcare (bounded).  The adapters here exist to show that
the typed adapter protocol can be implemented for additional domains, and to
keep older benchmark wrappers under ``orius.orius_bench`` and
``orius.universal_framework`` importable.

Do not cite numbers from this package in publication-facing artifacts.  These
adapters are not exercised by ``make paper-verify``, are not covered by the
promoted runtime governance surface, and are not included in the SHA-256
artifact manifests under ``reports/``.
"""

__all__: list[str] = []
