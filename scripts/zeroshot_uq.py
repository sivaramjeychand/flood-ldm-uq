"""
Zero-shot UQ transfer: the frozen Wollombi LDM on another catchment, with and without the stem
gain, scored as decoder_uq / stem_gain_probe score Wollombi.

Methods (--methods, each NAME=SPEC or just SPEC):
  base                  the frozen LDM, no gain (decoder_uq's `base`)
  uniform:SIGMA         untrained per-draw stem gain 1 + SIGMA * eps (stem_gain_probe at SIGMA)
  adapter:PATH          a trained stem adapter checkpoint (train_stem_adapter.py results/stem_adapter.pt),
                        any mode, including spatial
All methods go through the StemAdapter forward hook (scaling the stem's output channel = the probe's
weight-row scaling) and share one eps vector per draw (seed + 3) and the same noise seeds, so they
differ only through the gain.

Scenes. Target (the -c config, e.g. config/wollombi-zeroshot-burnett.json): the first-N-files
protocol of the Wollombi scripts picks one arbitrary (os.listdir-order) stretch, which on a mostly
dry catchment can have no flooded pixels at all. Instead: sorted filenames, seeded shuffle, keep
scenes whose flooded fraction (HR > --flood_threshold_cm) is >= --min_flooded_frac, first
--n_eval to eval and the next --n_calib to calib. Source (--source_config, Wollombi): exactly the
decoder_uq / stem_gain_probe calib set and seed, so the source quantiles match those runs.

Rows per method, all on the target's flooded eval pixels:
  raw        the ensemble's own central interval. The zero-shot number: nothing is refit.
  cqr_src    CQR fit on the Wollombi calib set, applied to the target. Zero-shot conformal: its
             coverage guarantee assumes exchangeability, which a catchment shift breaks.
  scale_src  the ensemble's own interval scaled by a factor fit on Wollombi calib.
  cqr_tgt    CQR fit on target calib scenes. Not zero-shot (needs target labels); the recalibrated
             reference.
  scale_tgt  the scaled interval fit on target calib.

Usage (from the repo root):
    python scripts/zeroshot_uq.py -c config/wollombi-zeroshot-burnett.json -output_dir experiments/zeroshot_uq/burnett \
        --methods base uniform:0.065 uniform:0.075 v2=adapter:experiments/stem_adapter/v2-sigma-is/results/stem_adapter.pt
    # plumbing check:
    python scripts/zeroshot_uq.py -c config/wollombi-zeroshot-burnett.json -output_dir experiments/zeroshot_uq/smoke \
        --n_eval 4 --n_calib 4 --n_samples 6 --methods base uniform:0.075
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

from decoder_uq import conformal, ensemble_metrics
from stem_gain_probe import scaled_conformal


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Target (zero-shot) config')
    parser.add_argument('-output_dir', type=str, default='experiments/zeroshot_uq')
    parser.add_argument('--source_config', type=str, default='config/wollombi-trnf.json',
                        help='Source-catchment config for the *_src conformal rows')
    parser.add_argument('--methods', type=str, nargs='+', default=['base', 'uniform:0.075'],
                        help='NAME=SPEC or SPEC; SPEC is base, uniform:SIGMA or adapter:PATH')
    parser.add_argument('--n_eval', type=int, default=40)
    parser.add_argument('--n_calib', type=int, default=40)
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--min_flooded_frac', type=float, default=0.01,
                        help='Target scenes need at least this fraction of pixels above --flood_threshold_cm')
    parser.add_argument('--max_screen', type=int, default=20000, help='Target scenes to screen at most')
    parser.add_argument('--ci', type=float, default=0.90)
    parser.add_argument('--flood_threshold_cm', type=float, default=5.0)
    parser.add_argument('--cp_eps_cm', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-debug', '-d', action='store_true')
    args = parser.parse_args()
    args.phase = 'test'
    return args


def parse_method(s):
    name, spec = s.split('=', 1) if '=' in s else (s, s)
    kind, _, val = spec.partition(':')
    if kind not in ('base', 'uniform', 'adapter') or (kind != 'base' and not val):
        raise SystemExit(f'bad method {s!r}: use base, uniform:SIGMA or adapter:PATH (optionally NAME=...)')
    if kind == 'adapter' and not os.path.isfile(val):
        raise SystemExit(f'adapter checkpoint not found: {val}')
    return name, kind, val


def read_config(path):
    """JSON with // comments, as core.logger.parse reads it."""
    with open(path) as f:
        return json.loads(''.join(line.split('//')[0] + '\n' for line in f))


def main():
    import core.logger as Logger
    import data as Data
    import data.util as Util
    import model as Model
    from collapse_ablation import Runner
    from train_stem_adapter import StemAdapter

    args = parse_args()
    methods = [parse_method(s) for s in args.methods]  # fail on a bad spec before loading anything
    opt = Logger.dict_to_nonedict(Logger.parse(args))
    logger = Logger.setup_logger('base', opt['path']['log'], 'zeroshot_uq', screen=True)
    logger.info(Logger.dict2str(opt))
    logger.info(f'zeroshot_uq args: {vars(args)}')
    torch.backends.cudnn.benchmark = True
    if opt['model']['which_model_G'] != 'ddpm':
        raise SystemExit('zeroshot_uq needs a latent (ddpm) config')

    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['test'], schedule_phase='test')
    runner = Runner(diffusion, opt, diffusion.device)
    dev = diffusion.device
    bs_test = runner.resolve_bs(opt['model']['beta_schedule']['test'].get('backward_start') or 'full')

    meta = opt['datasets']['meta']
    src = read_config(args.source_config)
    for k in ('max_depth', 'dem_max_value', 'dem_min_value', 'norm_min', 'norm_max'):
        if src['datasets']['meta'][k] != meta[k]:
            raise SystemExit(f'source and target normalisation differ on {k}; the model must see Wollombi scaling')
    norm_range = (meta['norm_min'], meta['norm_max'])
    cm_per_unit = meta['max_depth'] / (norm_range[1] - norm_range[0])
    to_cm = lambda x: (x.float().cpu().clamp(*norm_range) - norm_range[0]) * cm_per_unit

    # ---------------------------------------------------------- scenes
    tgt_set = Data.create_dataset(dict(opt['datasets']['test'], data_len=-1), 'test', meta, latent=False, dem=opt['dem'])
    order = sorted(range(len(tgt_set)), key=lambda i: tgt_set.filenames[i])
    order = [order[i] for i in np.random.default_rng(args.seed).permutation(len(order))]
    need, picked, frac_all, t0 = args.n_eval + args.n_calib, [], [], time.time()
    for i in order[:args.max_screen]:
        hr = Util.load_tiff(os.path.join(tgt_set.high_res_folder, tgt_set.filenames[i]), meta['max_depth'], norm=norm_range)
        f = (to_cm(hr) > args.flood_threshold_cm).float().mean().item()
        frac_all.append(f)
        if f >= args.min_flooded_frac:
            picked.append(i)
            if len(picked) == need:
                break
    logger.info(f'target screening: {len(frac_all)} scenes read ({time.time() - t0:.0f}s), '
                f'{np.mean(np.asarray(frac_all) >= args.min_flooded_frac):.1%} pass flooded fraction >= '
                f'{args.min_flooded_frac}; mean flooded fraction of all screened {np.mean(frac_all):.2%}')
    if len(picked) < need:
        raise SystemExit(f'only {len(picked)} target scenes pass the flood screen, need {need}; '
                         'lower --min_flooded_frac or raise --max_screen')
    ev_idx, ca_idx = picked[:args.n_eval], picked[args.n_eval:]

    # source calib: decoder_uq / stem_gain_probe protocol on the source test set
    src_set = Data.create_dataset(dict(src['datasets']['test'], data_len=-1), 'test', meta, latent=False, dem=opt['dem'])
    rng = np.random.default_rng(args.seed)
    rest = np.arange(min(args.n_eval, len(src_set)), len(src_set))
    src_idx = sorted(rng.choice(rest, size=min(args.n_calib, len(rest)), replace=False).tolist())

    def one_batch(ds, idx):
        return next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=len(idx),
                                                     shuffle=False, num_workers=2, pin_memory=True)))

    sets = {}
    for key, ds, idx, seed in (('eval', tgt_set, ev_idx, args.seed), ('calib_tgt', tgt_set, ca_idx, args.seed + 2),
                               ('calib_src', src_set, src_idx, args.seed + 2)):
        b = one_batch(ds, idx)
        y = to_cm(b['HR'])
        m = y > args.flood_threshold_cm
        sets[key] = {'batch': b, 'mask': m, 'y': y[m], 'seed': seed}
        logger.info(f'{key}: {len(idx)} scenes, flooded {m.float().mean():.1%} ({int(m.sum())} px)')

    results = {'args': vars(args), 'config': args.config, 'source_config': args.source_config,
               'backward_start': bs_test, 'methods': {n: f'{k}:{v}' if v else k for n, k, v in methods},
               'files': {k: list(s['batch']['filename']) for k, s in sets.items()}, 'eval': {}}
    out_dir = opt['path']['results']
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'zeroshot_uq.json')

    def dump():
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
    dump()

    # ---------------------------------------------------------- methods
    stem = runner.net.denoise_fn.downs[0]
    x_ch = opt['model']['unet']['out_channel']
    eps = torch.randn(args.n_samples, stem.out_channels, generator=torch.Generator().manual_seed(args.seed + 3))

    def build(kind, val):
        if kind == 'base':
            return None
        if kind == 'uniform':
            return StemAdapter(stem, 'stochastic', float(val), x_channels=x_ch).to(dev)
        ck = torch.load(val, map_location='cpu')
        a = ck['args']
        ad = StemAdapter(stem, ck['mode'], a.get('init_sigma', 0.05), x_channels=x_ch,
                         map_channels=a.get('map_channels', 1), max_sigma=a.get('max_sigma', 0.5))
        ad.load_state_dict(ck['state_dict'])
        return ad.to(dev)

    @torch.no_grad()
    def ensemble(adapter, s):
        """(K, P) flooded-pixel ensemble, seeded once per set as decoder_uq / stem_gain_probe."""
        b = s['batch']
        lr, d = b['SR'].to(dev), b['DEM'].to(dev) if 'DEM' in b else None
        torch.manual_seed(s['seed'])
        out = []
        for k in range(args.n_samples):
            if adapter is not None:
                adapter.current = (adapter.gains(eps[k:k + 1].to(dev)).expand(lr.shape[0], -1), adapter.beta)
            out.append(to_cm(runner.decode(runner.sample_latent(lr, d, bs_test, 'sample')))[s['mask']])
            if adapter is not None:
                adapter.current = None
        return torch.stack(out, 0)

    ev = sets['eval']
    for name, kind, val in methods:
        t0 = time.time()
        adapter = build(kind, val)
        spatial = adapter is not None and adapter.mode == 'spatial'
        ens, sigma_stats = {}, {}
        for k in ('eval', 'calib_src', 'calib_tgt'):
            if spatial:
                adapter.reset_stats()
            ens[k] = ensemble(adapter, sets[k])
            if spatial:
                sigma_stats[k] = adapter.summary()
        E, Cs, Ct = ens['eval'], ens['calib_src'], ens['calib_tgt']
        ys, yt = sets['calib_src']['y'], sets['calib_tgt']['y']
        r = {'raw': ensemble_metrics(E, ev['y'], args.ci),
             'cqr_src': conformal(Cs, ys, E, ev['y'], args.ci, 'cqr', args.cp_eps_cm),
             'scale_src': scaled_conformal(Cs, ys, E, ev['y'], args.ci, args.cp_eps_cm),
             'cqr_tgt': conformal(Ct, yt, E, ev['y'], args.ci, 'cqr', args.cp_eps_cm),
             'scale_tgt': scaled_conformal(Ct, yt, E, ev['y'], args.ci, args.cp_eps_cm),
             'raw_calib_src': ensemble_metrics(Cs, ys, args.ci),
             'adapter': adapter.summary() if adapter is not None else None,
             'sigma_map': sigma_stats or None,  # spatial: per scene set
             'seconds': time.time() - t0}
        if adapter is not None:
            adapter.handle.remove()
        results['eval'][name] = r
        dump()
        logger.info(f'[{name}] {results["methods"][name]} ({r["seconds"]:.0f}s); on Wollombi calib: '
                    f'raw cov {r["raw_calib_src"]["coverage"]:.3f} ss {r["raw_calib_src"]["spread_skill_ratio"]:.3f}')
        for k, a in sigma_stats.items():
            logger.info(f'[{name}] sigma map on {k}: mean {a["sigma_mean"]:.4f} range [{a["sigma_min"]:.4f}, '
                        f'{a["sigma_max"]:.4f}], within-scene pixel std {a["sigma_pix_std"]:.4f}')

    # ---------------------------------------------------------- table
    logger.info('=' * 78)
    logger.info(f'target {opt["datasets"]["test"]["catchment"]}: {len(ev_idx)} eval scenes, flooded pixels only, '
                f'K={args.n_samples}, nominal {args.ci}')
    logger.info(f"{'method':>12} {'row':>9} {'cov':>6} {'width':>7} {'IS':>7} {'CRPS':>6} {'RMSE':>7} {'ss':>6}")
    for name, r in results['eval'].items():
        for row in ('raw', 'cqr_src', 'scale_src', 'cqr_tgt', 'scale_tgt'):
            m = r[row]
            crps = f"{m['crps_cm']:6.3f}" if 'crps_cm' in m else f"{'-':>6}"
            ss = f"{m['spread_skill_ratio']:6.3f}" if 'spread_skill_ratio' in m else f"{'-':>6}"
            logger.info(f"{name:>12} {row:>9} {m['coverage']:6.3f} {m['mean_pi_width_cm']:7.3f} "
                        f"{m['interval_score_cm']:7.3f} {crps} {m['ensemble_mean_rmse_cm']:7.3f} {ss}")
    logger.info(f'Saved {out_path}')


if __name__ == '__main__':
    main()
