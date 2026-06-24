#!/usr/bin/env python3
"""
ReverseSDE / EnSF inflation tuning sweep.

This is the score-filter counterpart to letkf_r_tuning.py. It reuses the same
submit/organize/collect engine (imported from letkf_r_tuning) but ships EnSF
defaults:

  - template       ensf_runner_nonlinear.csv (method ReverseSDE)
  - --infla-values 0.6,0.8,1.0,1.2,1.4 (centered at 1; per Hristo's guidance,
                   the score filter already restores ensemble spread to the
                   prior std, so the swept factor need not be > 1)
  - --r-values     1 (held fixed, see below)
  - campaign root  reversesde_tuning_runs/<name>/
  - default name   ensf

Only inflation is swept. Unlike the LETKF, the localization radius r does not
affect the ReverseSDE analysis (it only appears in the run-folder name and a
harmless compute_sub_domains call), so r is held fixed at a single value rather
than swept. The inflation parameter tunes only the final multiplicative factor
applied after spread restoration (sequential_methods.py ReverseSDE:
covariance_inflation = xbar + infla * X'). You can still pass --r-values
explicitly if you ever want to vary it.

Examples
--------
    python reversesde_tuning.py submit --template ensf_runner_wdg_wsg.csv --infla-values 0.6,0.8,1.0,1.2,1.4 --exp-settings ../LETKF_tuning/t21_80_0.05_30/ --name wdg_wsg_inflation

    python reversesde_tuning.py collect reversesde_tuning_runs/ensf/manifest.json
"""

from letkf_r_tuning import build_parser


def main():
    parser = build_parser(
        default_template="ensf_runner_nonlinear.csv",
        default_infla="0.6,0.8,1.0,1.2,1.4",
        default_r="1",
        default_name="ensf",
        campaign_root="reversesde_tuning_runs",
    )
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
