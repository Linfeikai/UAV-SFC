# Frozen UAV-SFC deployment scorer release

All three independent training seeds passed the agreed original-setting criteria on the preregistered 50-episode confirmation and 100-episode fresh cohort. See final_audit.json for every metric, hash and measured cost.

The primary model remains model_seed42.pt, frozen before independent replicas. Models 43/44 are from-scratch reproductions with separately generated data; no ensemble or post-test seed selection is used.

This network learns selection among eight whole-deployment proposals from the existing observation. Shared analytical velocity braking supplies motion; this is not a learned mobility result. The simulator is used only to generate training labels, never at inference.

Environment: original heterogeneous_hybrid, Poisson mean .5 per UE per slot, CPU 500MHz, 4 UAV /20 UE /M12 /K6 /40 steps; reward, battery, collision and bandwidth unchanged. fixed_env_config.yaml is the exact audited physical-equivalent YAML backup. Full flattened equality is checked by scientific_source/experiments/server_20261021/fixed_hybrid_environment_manifest.json.

Runtime for all three released models: Python3.10.19 /PyTorch2.11.0+cpu /NumPy2.2.6. Existing sfc_env environment also provides the remaining project dependencies. No CUDA fallback was used.

Inference from the project source root:

```python
from algos.placement_scorer import ScoredDeploymentPolicy
policy = ScoredDeploymentPolicy.load("model_seed42.pt", device="cpu")
action, _ = policy.predict(observation, deterministic=True)
```

The source snapshot and SOURCE_COMMIT are supplied for isolated evaluation. From this release directory in PowerShell, a further untouched cohort can be evaluated using:

```powershell
& 'C:/Users/30395/.conda/envs/sfc_env/python.exe' scientific_source/scripts/evaluate_scorer_cohort.py --root scientific_source --config fixed_env_config.yaml --model model_seed42.pt --expected-sha256 1b26b5f4d658bfc30d148fb159f3aec70dcf6c5e2170c05ee816e5c60bc3ad0e --run-id additional_check --seed 94000 --episodes 100 --output additional_check_94000
```

The output folder must not already exist. This optional command has not been run and is not part of the reported qualification. The report uses confirmation91000–91049 and fresh92000–92099, matched across training seeds and comparators.

Full training data and logs are separately preserved in scorer_scale1000_seed42_evidence.tar and scorer_scale1000_replicates_43_44_evidence.tar with checksum-verified archive manifests. Actual run IDs are local experiment identities; W&B publication is not claimed.
