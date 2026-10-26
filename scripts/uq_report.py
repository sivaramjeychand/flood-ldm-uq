"""
Offline report on stem_gain_probe sweeps and train_stem_adapter runs (no GPU, reads the JSONs).

1. Held-out sigma choice. Picks the uniform sigma on the calib scenes (`raw_calib`, disjoint from
   eval) by each criterion below, and reports that sigma's eval row next to the eval-chosen
   (oracle) one. These are the uniform bars that are fair to quote.
     best IS      min raw interval score on calib
     best CRPS    min raw CRPS on calib
     ss = 1       spread/skill closest to 1 on calib
     cov >= ci    smallest sigma whose calib coverage reaches --ci (the "free" 0.9 baseline)
2. Adapter vs uniform at equal spread/skill, with a paired bootstrap over eval scenes. Every
   resample draws the same scenes for the adapter and every uniform sigma, rebuilds the uniform
   coverage/IS curve from the per-scene sums, and interpolates it at the adapter's own pooled
   spread/skill. Needs `per_scene` in both files (scripts updated 2026-10-26) and the same
   --seed / --n_eval / --n_samples, so the eval scenes and noise match.

Usage (from the repo root):
    python scripts/uq_report.py --sweep experiments/stem_gain/s0-dense/results/stem_gain_probe.json \
        --adapter experiments/stem_adapter/v3-spatial-is-s0/results/train_stem_adapter.json \
                  experiments/stem_adapter/v3-nomap-s0/results/train_stem_adapter.json
"""
import argparse
import json

import numpy as np

KEYS = ('n_pixels', 'covered', 'interval_score_cm', 'ens_var', 'sq_err')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', nargs='+', required=True, help='stem_gain_probe.json files (merged)')
    parser.add_argument('--adapter', nargs='*', default=[],
                        help='train_stem_adapter.json files; append :tag for one eval (default every tag)')
    parser.add_argument('--ci', type=float, default=0.90)
    parser.add_argument('--n_boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def run_key(a):
    return a['seed'], a['n_eval'], a['n_samples']


def pooled(ps, idx):
    """Coverage, IS and spread/skill pooled over scenes idx (with repeats) from per-scene sums."""
    s = {k: np.asarray(ps[k], dtype=float)[idx].sum(-1) for k in KEYS}
    return {'coverage': s['covered'] / s['n_pixels'], 'interval_score_cm': s['interval_score_cm'] / s['n_pixels'],
            'ss': np.sqrt(s['ens_var'] / s['sq_err'])}


def interp_at(ss_curve, val_curve, ss):
    """Per resample: val interpolated on the uniform curve at ss; nan outside the curve's range."""
    out = np.full(ss.shape, np.nan)
    for b in range(ss.shape[0]):
        o = np.argsort(ss_curve[:, b])
        x, v = ss_curve[o, b], val_curve[o, b]
        if x[0] <= ss[b] <= x[-1]:
            out[b] = np.interp(ss[b], x, v)
    return out


def main():
    args = parse_args()
    rows, key = {}, None
    for p in args.sweep:
        with open(p) as f:
            d = json.load(f)
        k = run_key(d['args'])
        if key is not None and k != key:
            raise SystemExit(f'{p}: seed/n_eval/n_samples {k} differ from {key}; one seed per report')
        key = k
        rows.update({float(s): r for s, r in d['sweep'].items()})
    sigmas = sorted(rows)
    print(f'uniform sweep: seed {key[0]}, {key[1]} eval scenes, K={key[2]}, sigmas {sigmas}')

    # ------------------------------------------------------------- 1. held-out sigma choice
    cal = [s for s in sigmas if 'raw_calib' in rows[s]]
    if len(cal) < len(sigmas):
        print(f'no raw_calib for sigmas {sorted(set(sigmas) - set(cal))} (old sweep); held-out choice uses the rest')
    crit = {'best IS': lambda r: r['interval_score_cm'], 'best CRPS': lambda r: r['crps_cm'],
            'ss = 1': lambda r: abs(r['spread_skill_ratio'] - 1)}
    print(f"\n{'criterion':>10} {'chosen on':>9} {'sigma':>6} {'cov':>6} {'width':>7} {'IS':>7} {'CRPS':>6} {'RMSE':>7} {'ss':>6}")

    def line(name, where, s):
        r = rows[s]['raw']
        print(f"{name:>10} {where:>9} {s:6g} {r['coverage']:6.3f} {r['mean_pi_width_cm']:7.2f} "
              f"{r['interval_score_cm']:7.2f} {r['crps_cm']:6.3f} {r['ensemble_mean_rmse_cm']:7.3f} "
              f"{r['spread_skill_ratio']:6.3f}")
    for name, f in crit.items():
        if cal:
            line(name, 'calib', min(cal, key=lambda s: f(rows[s]['raw_calib'])))
        line(name, 'eval', min(sigmas, key=lambda s: f(rows[s]['raw'])))
    reach = [s for s in cal if rows[s]['raw_calib']['coverage'] >= args.ci]
    if reach:
        line(f'cov>={args.ci:g}', 'calib', reach[0])
    elif cal:
        print(f'{"cov>=" + format(args.ci, "g"):>10}     calib  none (max calib coverage '
              f"{max(rows[s]['raw_calib']['coverage'] for s in cal):.3f})")

    # ------------------------------------------------------------- 2. adapter vs uniform, bootstrap
    ps_sig = [s for s in sigmas if 'per_scene' in rows[s]]
    if not args.adapter:
        return
    if len(ps_sig) < 2:
        raise SystemExit('need per_scene for >= 2 uniform sigmas to compare adapters; re-run the sweep')
    n = len(rows[ps_sig[0]]['per_scene']['n_pixels'])
    rng = np.random.default_rng(args.seed)
    idx = np.vstack([np.arange(n), rng.integers(0, n, size=(args.n_boot, n))])  # row 0 = the actual scenes
    uni = [pooled(rows[s]['per_scene'], idx) for s in ps_sig]
    u_ss = np.stack([u['ss'] for u in uni])
    print(f'\nadapter - uniform at the adapter\'s spread/skill; {n} eval scenes, {args.n_boot} paired resamples, '
          f'95% percentile CI; uniform sigmas {ps_sig}')
    print(f"{'run [tag]':>34} {'ss':>6} {'d cov':>7} {'95% CI':>17} {'P>0':>5} {'d IS':>7} {'95% CI':>17} {'P<0':>5}")
    for spec in args.adapter:
        path, sep, only = spec.rpartition(':')
        if not sep or only.endswith('.json') or '/' in only or '\\' in only:  # no tag (or a drive letter)
            path, only = spec, ''
        with open(path) as f:
            d = json.load(f)
        if run_key(d['args']) != key:
            print(f'{path}: seed/n_eval/n_samples {run_key(d["args"])} != sweep {key}; skipped')
            continue
        for tag, r in d['eval'].items():
            if (only and tag != only) or 'per_scene' not in r:
                continue
            a = pooled(r['per_scene'], idx)
            out = []
            for k, sign in (('coverage', 1), ('interval_score_cm', -1)):
                diff = a[k] - interp_at(u_ss, np.stack([u[k] for u in uni]), a['ss'])
                boot = diff[1:][np.isfinite(diff[1:])]
                lo, hi = np.percentile(boot, [2.5, 97.5]) if boot.size else (np.nan, np.nan)
                out.append((diff[0], lo, hi, (sign * boot > 0).mean() if boot.size else np.nan, boot.size))
            name = f"{d['args']['output_dir'].rstrip('/').split('/')[-1]} [{tag}]"
            (dc, cl, ch, pc, nb), (di, il, ih, pi, _) = out
            print(f"{name[-34:]:>34} {a['ss'][0]:6.3f} {dc:+7.3f} [{cl:+7.3f}, {ch:+7.3f}] {pc:5.2f} "
                  f"{di:+7.2f} [{il:+7.2f}, {ih:+7.2f}] {pi:5.2f}"
                  + (f'  ({args.n_boot - nb} resamples outside the curve)' if nb < args.n_boot else ''))


if __name__ == '__main__':
    main()
