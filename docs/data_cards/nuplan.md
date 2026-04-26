# nuPlan Data Card

## Role In ORIUS

nuPlan is the active autonomous-vehicle evidence source for the public ORIUS claim
surface. The paper uses it only as a bounded replay row for ego-speed and
relative-gap monitoring under the TTC plus predictive-entry-barrier contract.

## Claim Boundary

- Defended: bounded replay conversion, deterministic split assignment, conformal
  feature generation, runtime certificates, and CertOS audit verification.
- Not defended: full closed-loop road deployment, policy optimality, lane-change
  planning, or full autonomy closure.

## Source And Manifest Policy

Raw nuPlan archives are not tracked in Git. The ingestion path records completed
archives in `nuplan_source_manifest.json`, skips partial `.crdownload` files, and
excludes map-only archives from training inputs. Scenario identifiers retain archive
identity so multiple completed archives cannot collide.

## Active Artifact Surface

The bounded report surface is `reports/orius_av/nuplan_bounded/`. The current
public manuscript cites the bounded denominator reported there: 10 runtime
scenarios, 410 runtime steps, and 410 CertOS-verified ORIUS certificates.

