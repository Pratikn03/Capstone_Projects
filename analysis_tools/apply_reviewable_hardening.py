"""Apply a reviewable patch only to an ephemeral audit checkout; preserve before/after diff."""
from pathlib import Path
import hashlib, shutil, difflib, json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit_output/hardening';OUT.mkdir(parents=True,exist_ok=True)
p=ROOT/'services/api/routers/dc3s.py';before=p.read_text()
expected='9da959357657fe8c6b69ff0313cc374c3182c550eeb3e4670ccdde880b405a04'
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,'Pinned source differs; refusing an unreviewed patch'
after=before
old='from orius.universal_theory.battery_instantiation import (\n    certificate_expiration_bound,\n    certificate_validity_horizon,\n)'
assert old in after
after=after.replace(old,'from orius.dc3s.release_contract import assess_battery_release')
old='    current_soc_mwh: float = Field(..., ge=0.0)'
assert old in after
after=after.replace(old,'    current_soc_mwh: float = Field(..., ge=0.0, allow_inf_nan=False, strict=True)\n    state_error_bound_mwh: float | None = Field(default=None, ge=0.0, allow_inf_nan=False, strict=True)\n    transition_error_bound_mwh: float | None = Field(default=None, ge=0.0, allow_inf_nan=False, strict=True)')
old='        renewables = wind_yhat + solar_yhat\n'
assert old in after
after=after.replace(old,old+'        for signal_name, values in (("load", load_yhat), ("renewables", renewables)):\n            if not np.isfinite(values).all() or np.any(values < 0):\n                raise HTTPException(status_code=422, detail=f"{signal_name} forecast is nonfinite or negative; no silent correction was applied")\n')
old='        ftit_state = update_ftit_state(\n'
assert after.count(old)==1
after=after.replace(old,'        if not min_soc - 1e-12 <= float(req.current_soc_mwh) <= max_soc + 1e-12:\n            raise HTTPException(status_code=422, detail="Reported energy is outside the configured analysis envelope")\n'+old)
start=after.index('        # T5/T6 runtime expiry check')
end=after.index('        runtime_mode = _resolve_iot_mode',start)
block='''        # Explicit energy contract: MW forecast residuals are never used as MWh state bounds.
        # The caller-supplied bounds remain premises. This only checks one timestep;
        # it does not claim a multi-step T5/T6 expiration law or independent validation.
        try:
            state_contract = assess_battery_release(
                current_soc_mwh=float(req.current_soc_mwh), action=safe_action,
                constraints=constraints, state_error_bound_mwh=req.state_error_bound_mwh,
                transition_error_bound_mwh=req.transition_error_bound_mwh,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        uncertainty_meta["physical_state_contract"] = state_contract
        uncertainty_meta["validity_horizon_tau_t"] = state_contract["validity_horizon_steps"]
        uncertainty_meta["expiry_lower_bound"] = None
        if not state_contract["supported"]:
            guarantee_fail_reasons = list(guarantee_fail_reasons or []) + list(state_contract["reasons"])
            guarantee_passed = False

'''
after=after[:start]+block+after[end:]
after=after.replace('        if runtime_mode == "active" and not guarantee_passed:','        if (runtime_mode == "active" or req.enqueue_iot) and not guarantee_passed:')
p.write_text(after)
helper=ROOT/'src/orius/dc3s/release_contract.py';shutil.copyfile(ROOT/'analysis_tools/release_contract.py',helper)
diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/services/api/routers/dc3s.py',tofile='b/services/api/routers/dc3s.py'))
diff+=''.join(difflib.unified_diff([],helper.read_text().splitlines(True),fromfile='/dev/null',tofile='b/src/orius/dc3s/release_contract.py'))
(OUT/'reviewable_hardening.patch').write_text(diff)
(OUT/'source_change.json').write_text(json.dumps({'original_router_sha256':expected,'patched_router_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'changes':['reject invalid forecasts instead of uncaught solver error','explicit out-of-envelope HTTP422','require declared energy and transition error for active/queued release','no load-MW-as-SOC-MWh horizon','only conditional one-step horizon; T6 not asserted'],'public_data_bounds_validated':False,'main_branch_modified':False,'intentionally_breaks_legacy_unbounded_enqueue':True},indent=2))
print('PATCH_APPLIED_TO_DISPOSABLE_CHECKOUT',str(OUT/'reviewable_hardening.patch'))
