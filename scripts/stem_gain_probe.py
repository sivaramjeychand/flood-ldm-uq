"""
Stochastic per-channel stem gain: an untrained check on whether per-member randomness in the
U-Net stem can buy coverage more cheaply than split conformal.

Each ensemble draw k runs the reverse chain with its own gain on the output channels of the
stem conv (downs[0], 64 channels): gamma_m = 1 + sigma * eps_m, eps_m ~ N(0, I), member
m = k mod M. The same gamma is applied at every reverse step of that draw. Nothing is trained;
this is the inference-only rung below a trained stem SSF.

cond_scale_probe.py (probe B) applied one gain to every draw, so it only measured spread
across noise seeds. Here the gain differs between draws, so between-member variation is part
of the ensemble.

Scenes, seeds and scoring match decoder_uq.py: eval = the first --n_eval test scenes, calib =
the same random --n_calib test scenes, flooded pixels only. At sigma=0 the ensemble is
decoder_uq's `base`, so its cp_cqr row reproduces the bar to beat (IS 51.454cm in v1).
The eps_m are fixed across sigma and every sigma replays the same noise stream (the number
of random draws doesn't depend on the weights), so the curve differs only through sigma.

Rows per sigma
  raw       the ensemble's own central interval. Coverage moves with sigma.
  cp_cqr    CQR on this ensemble: additive widening to nominal coverage on calib.
  cp_scale  the ensemble's own interval scaled about its midpoint by one factor fit on calib.
            It keeps the ensemble's per-pixel shape, so it shows whether that shape tracks
            the error better than base does.
  raw_calib the raw row on the calib scenes. Calib is disjoint from eval, so choosing sigma on
            raw_calib and reporting it on raw is a held-out choice (scripts/uq_report.py).
Each sigma also stores `per_scene`: per-eval-scene sums for the raw ensemble (pixels, covered,
interval score, ensemble variance, squared error of the mean), so uq_report.py can bootstrap
over scenes, paired with an adapter run on the same eval scenes.
--k_curve K ...: the raw eval row recomputed on random K-member subsets of the draws (mean of --k_reps),
with the ensemble-mean RMSE next to the oracle per-scene best member and the mean single-member RMSE
(`k_curve`, also used by train_stem_adapter.py). No extra sampling; at K = 1 the interval is a point.
Gate: training a stem adapter is only worth it if some sigma > 0 beats the sigma=0 cp_cqr
interval score at matched (calibrated) coverage.

Usage (from the repo root):
    python scripts/stem_gain_probe.py -c config/wollombi-trnf.json -output_dir experiments/stem_gain/v1
    # plumbing check:
    python scripts/stem_gain_probe.py -c config/wollombi-trnf.json -output_dir experiments/stem_gain/smoke \
        --n_eval 4 --n_calib 4 --n_samples 6 --sigmas 0 0.01
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

import numpy as np
import torch

from decoder_uq import _conformal_q, conformal, ensemble_metrics, interval_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-output_dir', type=str, default='experiments/stem_gain')
    parser.add_argument('--n_eval', type=int, default=40)
    parser.add_argument('--n_calib', type=int, default=40)
    parser.add_argument('--n_samples', type=int, default=50, help='Draws per scene at every sigma')
    parser.add_argument('--n_members', type=int, default=0,
                        help='Distinct gain vectors; draw k uses member k mod M. 0 = one per draw')
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0.0, 0.005, 0.01, 0.02, 0.03],
                        help='Gain std values. Keep 0 in the list: it is the base ensemble and sets the bar')
    parser.add_argument('--ci', type=float, default=0.90)
    parser.add_argument('--flood_threshold_cm', type=float, default=5.0)
    parser.add_argument('--cp_eps_cm', type=float, default=0.5, help='Floor on the half-width in the cp_scale score')
    parser.add_argument('--k_curve', type=int, nargs='*', default=[],
                        help='Ensemble sizes for the raw-metrics-vs-K curve on eval (subsets of the --n_samples draws)')
    parser.add_argument('--k_reps', type=int, default=10, help='Random subsets per K in the K curve')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-debug', '-d', action='store_true')
    args = parser.parse_args()
    args.phase = 'test'
    return args


def scaled_conformal(cal_ens, cal_y, ev_ens, ev_y, ci, eps):
    """Split conformal that keeps the ensemble's own interval shape: score |y - mid| / half,
    so the calibrated interval is mid +- q * half (half floored at eps)."""
    alpha = 1 - ci

    def mid_half(e):
        lo, hi = torch.quantile(e, alpha / 2, dim=0), torch.quantile(e, 1 - alpha / 2, dim=0)
        return (lo + hi) / 2, (hi - lo) / 2 + eps
    cm, ch = mid_half(cal_ens)
    q = _conformal_q((cal_y - cm).abs() / ch, ci)
    em, eh = mid_half(ev_ens)
    out = interval_metrics((em - q * eh).clamp(min=0), em + q * eh, ev_y, ci)
    out.update({'q': q, 'ensemble_mean_rmse_cm': torch.sqrt(((ev_ens.mean(0) - ev_y) ** 2).mean()).item()})
    return out


def per_scene_sums(ens, y, sid, ci):
    """Per-scene sums for the raw (K, P) ensemble vs (P,) truth, sid (P,) the scene index of each
    pixel. Pooling any subset of scenes reproduces ensemble_metrics' coverage, IS and spread/skill."""
    alpha = 1 - ci
    lo, hi = torch.quantile(ens, alpha / 2, dim=0), torch.quantile(ens, 1 - alpha / 2, dim=0)
    score = (hi - lo) + (2 / alpha) * ((lo - y).clamp(min=0) + (y - hi).clamp(min=0))
    n = int(sid.max()) + 1 if sid.numel() else 0
    sums = lambda v: torch.bincount(sid, weights=v.double(), minlength=n).tolist()
    return {'n_pixels': torch.bincount(sid, minlength=n).tolist(),
            'covered': sums(((y >= lo) & (y <= hi)).float()),
            'interval_score_cm': sums(score),
            'ens_var': sums(ens.var(0, unbiased=ens.shape[0] > 1)),
            'sq_err': sums((ens.mean(0) - y) ** 2)}


def k_curve(ens, y, sid, ci, ks, reps, seed):
    """Raw metrics against ensemble size. For each k, the mean over `reps` random k-member subsets of
    the (K, P) ensemble (one subset when k = K, which reproduces the full row). Each row also has
    best_member_rmse_cm, the oracle best member: per scene, the member with the lowest flooded MSE,
    chosen with the truth, so it shows how close the ensemble gets, not a usable prediction. And
    member_rmse_cm, the mean single-member RMSE. Both are pooled over pixels like ensemble_mean_rmse_cm."""
    K, P = ens.shape
    sid = sid.to(ens.device)
    sq = torch.stack([torch.bincount(sid, weights=((e - y) ** 2).double(), minlength=int(sid.max()) + 1)
                      for e in ens]).cpu()  # (K, scenes) squared-error sums
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k in sorted({min(k, K) for k in ks}):
        rows = []
        for _ in range(1 if k == K else reps):
            idx = torch.randperm(K, generator=gen)[:k]
            m = ensemble_metrics(ens[idx.to(ens.device)], y, ci)
            m['best_member_rmse_cm'] = (sq[idx].min(0).values.sum() / P).sqrt().item()
            m['member_rmse_cm'] = (sq[idx].sum(1) / P).sqrt().mean().item()
            rows.append(m)
        out[str(k)] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        out[str(k)]['reps'] = len(rows)
    return out


def log_k_curve(logger, kc):
    logger.info(f"{'K':>4} {'cov':>6} {'width':>7} {'IS':>7} {'CRPS':>6} {'ss':>6} "
                f"{'ensmean':>8} {'best':>7} {'member':>7}  (RMSE cm; best = oracle per-scene best member)")
    for k, m in kc.items():
        logger.info(f"{k:>4} {m['coverage']:6.3f} {m['mean_pi_width_cm']:7.3f} {m['interval_score_cm']:7.3f} "
                    f"{m['crps_cm']:6.3f} {m['spread_skill_ratio']:6.3f} {m['ensemble_mean_rmse_cm']:8.3f} "
                    f"{m['best_member_rmse_cm']:7.3f} {m['member_rmse_cm']:7.3f}")


def main():
    import core.logger as Logger
    import core.metrics as Metrics
    import data as Data
    import model as Model
    from collapse_ablation import Runner

    args = parse_args()
    opt = Logger.dict_to_nonedict(Logger.parse(args))
    logger = Logger.setup_logger('base', opt['path']['log'], 'stem_gain_probe', screen=True)
    logger.info(Logger.dict2str(opt))
    logger.info(f'stem_gain_probe args: {vars(args)}')
    torch.backends.cudnn.benchmark = True
    if opt['model']['which_model_G'] != 'ddpm':
        raise SystemExit('stem_gain_probe needs a latent (ddpm) config, to match decoder_uq')

    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['test'], schedule_phase='test')
    runner = Runner(diffusion, opt, diffusion.device)
    dev = diffusion.device
    bs_test = runner.resolve_bs(opt['model']['beta_schedule']['test'].get('backward_start') or 'full')

    meta = opt['datasets']['meta']
    norm_range = (meta['norm_min'], meta['norm_max'])
    to_depth = lambda x: Metrics.unnormalize(x.float().cpu().clamp(*norm_range), meta['max_depth'], min_max=norm_range)

    # Same eval / calib scenes as decoder_uq.py (same rng, same draw order)
    base_opt = dict(opt['datasets']['test'], data_len=-1, use_shuffle=False)
    test_set = Data.create_dataset(base_opt, 'test', meta, latent=False, dem=opt['dem'])
    rng = np.random.default_rng(args.seed)
    eval_idx = list(range(min(args.n_eval, len(test_set))))
    rest = np.arange(len(eval_idx), len(test_set))
    calib_idx = sorted(rng.choice(rest, size=min(args.n_calib, len(rest)), replace=False).tolist())

    def one_batch(idx):
        return next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(test_set, idx), batch_size=len(idx),
                                                     shuffle=False, num_workers=2, pin_memory=True)))

    ev_batch, ca_batch = one_batch(eval_idx), one_batch(calib_idx)
    ev_hr, ca_hr = to_depth(ev_batch['HR']), to_depth(ca_batch['HR'])
    ev_mask, ca_mask = ev_hr > args.flood_threshold_cm, ca_hr > args.flood_threshold_cm
    y_ev, y_ca = ev_hr[ev_mask], ca_hr[ca_mask]
    ev_sid = torch.arange(ev_mask.shape[0]).view(-1, *[1] * (ev_mask.dim() - 1)).expand_as(ev_mask)[ev_mask]

    stem = runner.net.denoise_fn.downs[0]
    w0 = stem.weight.data.clone()
    b0 = stem.bias.data.clone() if stem.bias is not None else None
    M = args.n_members or args.n_samples
    eps = torch.randn(M, w0.shape[0], generator=torch.Generator().manual_seed(args.seed + 3))
    logger.info(f'stem conv {tuple(w0.shape)}, gain on {w0.shape[0]} output channels, {M} gain vectors; '
                f'eval {len(eval_idx)} scenes (flooded {ev_mask.float().mean():.1%}), '
                f'calib {len(calib_idx)} scenes, {args.n_samples} draws, backward_start={bs_test}')

    def set_gain(g):
        # Scaling the output channel = scaling that row of the weight and its bias
        if g is None:
            stem.weight.data.copy_(w0)
            if b0 is not None:
                stem.bias.data.copy_(b0)
        else:
            g = g.to(dev, w0.dtype)
            stem.weight.data.copy_(w0 * g.view(-1, 1, 1, 1))
            if b0 is not None:
                stem.bias.data.copy_(b0 * g)

    def ensemble(batch, mask, sigma, seed):
        """(K, P) flooded-pixel ensemble. Seeded once per set, as decoder_uq.draw_latents is."""
        lr, d = batch['SR'].to(dev), batch['DEM'].to(dev) if 'DEM' in batch else None
        torch.manual_seed(seed)
        out = []
        for k in range(args.n_samples):
            set_gain(None if sigma == 0 else 1 + sigma * eps[k % M])
            out.append(to_depth(runner.decode(runner.sample_latent(lr, d, bs_test, 'sample')))[mask])
        set_gain(None)
        return torch.stack(out, 0)

    results = {'args': vars(args), 'config': args.config, 'backward_start': bs_test,
               'eval_files': list(ev_batch['filename']), 'calib_files': list(ca_batch['filename']),
               'stem_channels': int(w0.shape[0]), 'n_gain_vectors': M, 'sweep': {}}
    out_dir = opt['path']['results']
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'stem_gain_probe.json')

    def dump():
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    ci, rows = args.ci, results['sweep']
    header = f"{'sigma':>6} {'method':>9} {'cov':>6} {'width':>7} {'IS':>7} {'CRPS':>6} {'RMSE':>7} {'ss':>6}"
    for sigma in args.sigmas:
        t0 = time.time()
        E = ensemble(ev_batch, ev_mask, sigma, args.seed)
        C = ensemble(ca_batch, ca_mask, sigma, args.seed + 2)  # independent noise from eval, as decoder_uq
        draw_rmse = torch.sqrt(((E - y_ev) ** 2).mean(1))
        r = {'raw': ensemble_metrics(E, y_ev, ci),
             'raw_calib': ensemble_metrics(C, y_ca, ci),
             'per_scene': per_scene_sums(E, y_ev, ev_sid, ci),
             'cp_cqr': conformal(C, y_ca, E, y_ev, ci, 'cqr', args.cp_eps_cm),
             'cp_scale': scaled_conformal(C, y_ca, E, y_ev, ci, args.cp_eps_cm),
             'per_draw_rmse_cm': {'mean': draw_rmse.mean().item(), 'min': draw_rmse.min().item(),
                                  'max': draw_rmse.max().item()},
             'seconds': time.time() - t0}
        if args.k_curve:
            r['k_curve'] = k_curve(E, y_ev, ev_sid, ci, args.k_curve, args.k_reps, args.seed + 4)
        rows[str(sigma)] = r
        dump()
        del E, C
        logger.info(f'sigma={sigma:g} done ({r["seconds"]:.0f}s); per-draw flooded RMSE '
                    f'{r["per_draw_rmse_cm"]["mean"]:.3f}cm (range {r["per_draw_rmse_cm"]["min"]:.3f}-'
                    f'{r["per_draw_rmse_cm"]["max"]:.3f})')
        logger.info(header)
        for name in ('raw', 'cp_cqr', 'cp_scale'):
            m = r[name]
            crps = f"{m['crps_cm']:6.3f}" if 'crps_cm' in m else f"{'-':>6}"
            ss = f"{m['spread_skill_ratio']:6.3f}" if 'spread_skill_ratio' in m else f"{'-':>6}"
            logger.info(f"{sigma:6g} {name:>9} {m['coverage']:6.3f} {m['mean_pi_width_cm']:7.3f} "
                        f"{m['interval_score_cm']:7.3f} {crps} {m['ensemble_mean_rmse_cm']:7.3f} {ss}")
        if 'k_curve' in r:
            log_k_curve(logger, r['k_curve'])

    # Gate: best calibrated interval score at sigma > 0 vs CQR on the base ensemble
    logger.info('=' * 78)
    if '0.0' in rows:
        bar = rows['0.0']['cp_cqr']['interval_score_cm']
        cands = [(rows[s][name]['interval_score_cm'], s, name)
                 for s in rows if float(s) > 0 for name in ('cp_cqr', 'cp_scale')]
        if cands:
            best_is, best_s, best_name = min(cands)
            results['gate'] = {'bar_cp_cqr_sigma0_is_cm': bar, 'best_is_cm': best_is, 'best_sigma': float(best_s),
                               'best_method': best_name, 'passes': best_is < bar}
            logger.info(f'GATE: best calibrated IS at sigma>0 is {best_is:.3f}cm ({best_name}, sigma={best_s}) '
                        f'vs sigma=0 cp_cqr {bar:.3f}cm -> {"PASS" if best_is < bar else "FAIL"}')
    else:
        logger.info('sigma=0 not in the sweep, so there is no in-run CQR bar (v1 decoder_uq: IS 51.454cm)')
    dump()
    logger.info(f'Saved {out_path}')


if __name__ == '__main__':
    main()
