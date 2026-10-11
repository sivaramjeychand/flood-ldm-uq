"""
Zero-training conditioning-gain probe.

Scales the conditioning slice W_c of the UNet stem conv (downs[0]) by a scalar s
at inference and sweeps s, recording flooded-pixel coverage, spread and
spread/skill against ensemble-mean RMSE. Because the stem conv is linear in its
input, W_c <- s * W_c is exactly equivalent to feeding s * [SR_latent, DEM_latent]
into the UNet; the noisy-latent slice and the bias are left alone.

Reading the result:
  - If no s in the sweep moves flooded-pixel spread meaningfully, a per-channel
    adapter on the same slice (SSF etc.) won't either -- the network is bypassing
    the noise deeper down, and the intervention has to go further in.
  - If spread does move, the best (coverage, RMSE) point on the curve is a rough
    ceiling for what an adapter on this slice can achieve.

Caveat: with backward_start set, the reverse chain is *initialised* from a lightly
noised SR latent (diffusion.py super_resolution). That pathway bypasses W_c
entirely, so it can pin samples to the coarse map regardless of s. Use
--backward_start to override it (e.g. 1000 = start from pure noise) and see
whether the init or the stem weights are what's holding spread down.

Every s value uses the same K seeds (common random numbers), so differences
between s values reflect s, not sampling noise.

Usage (run from repo root, same as coverage_diagnostic.py):
    python scripts/cond_scale_probe.py -c config/wollombi-trnf.json \
        -output_dir experiments/cond_scale_probe --n_scenes 10 --n_samples 20 \
        --scales 0 0.25 0.5 0.75 0.9 1.0 1.1 1.25 1.5
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
import torch.nn as nn

import core.logger as Logger
import core.metrics as Metrics
import data as Data
import model as Model
from coverage_diagnostic import coverage_and_stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-output_dir', type=str, default='experiments/cond_scale_probe')
    parser.add_argument('--n_scenes', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=20,
                        help='Stochastic draws per scene per scale value')
    parser.add_argument('--scales', type=float, nargs='+',
                        default=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5],
                        help='Values of s to multiply W_c by. 1.0 reproduces the checkpoint.')
    parser.add_argument('--target', choices=['cond', 'lr', 'dem'], default='cond',
                        help='Which input slice of the stem conv to scale: '
                             'cond = SR+DEM latents, lr = SR latent only, dem = DEM latent only')
    parser.add_argument('--backward_start', type=int, default=None,
                        help='Override the test schedule backward_start (truncated reverse diffusion). '
                             'Set to n_timestep (1000) to start from pure noise.')
    parser.add_argument('--ci', type=float, default=0.90)
    parser.add_argument('--flood_threshold_cm', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-debug', '-d', action='store_true')
    args = parser.parse_args()
    args.phase = 'test'  # required by core.logger.Logger.parse
    return args


def cond_slice(target, dem):
    # ddpm path concatenates [x_noisy, lr, dem], 4 latent channels each (diffusion.py super_resolution)
    if target == 'lr':
        return slice(4, 8)
    if target == 'dem':
        if not dem:
            raise ValueError('--target dem requires "dem": true in the config')
        return slice(8, 12)
    return slice(4, 12 if dem else 8)


def plot_sweep(rows, out_path, ci):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    s = [r['scale'] for r in rows]
    f = [r['flooded_pixels'] for r in rows]
    rmse = [x['ensemble_mean_rmse_cm'] for x in f]
    panels = [
        ('Flooded-pixel coverage', [x['coverage'] for x in f], ci),
        ('Flooded-pixel spread/skill', [x['spread_skill_ratio'] for x in f], 1.0),
        ('Flooded-pixel spread (cm)', [x['spread_cm'] for x in f], None),
    ]
    ink, muted, mark = '#1f2328', '#6e7781', '#2a6fdb'

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, y, ref) in zip(axes, panels):
        ax.plot(rmse, y, color=mark, lw=2, marker='o', ms=6, zorder=3)
        for xi, yi, si in zip(rmse, y, s):
            ax.annotate(f's={si:g}', (xi, yi), textcoords='offset points', xytext=(5, 5),
                        fontsize=8, color=muted)
        if ref is not None:
            ax.axhline(ref, color=muted, lw=1, ls='--', zorder=1)
        ax.set_title(title, color=ink, fontsize=10, loc='left')
        ax.set_xlabel('Ensemble-mean RMSE, flooded pixels (cm)', color=muted, fontsize=9)
        ax.grid(True, color='#d0d7de', lw=0.5)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=muted, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    logger = Logger.setup_logger('base', opt['path']['log'], 'cond_scale_probe', screen=True)
    logger.info(Logger.dict2str(opt))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    test_opt = dict(opt['datasets']['test'])
    test_opt['data_len'] = args.n_scenes
    test_opt['batch_size'] = args.n_scenes
    test_opt['use_shuffle'] = False
    test_set = Data.create_dataset(test_opt, 'test', opt['datasets']['meta'], latent=False, dem=opt['dem'])
    test_loader = Data.create_dataloader(test_set, test_opt, 'test')
    fixed_batch = next(iter(test_loader))
    n_scenes = fixed_batch['HR'].shape[0]

    schedule = dict(opt['model']['beta_schedule']['test'])
    if args.backward_start is not None:
        schedule['backward_start'] = None if args.backward_start >= schedule['n_timestep'] else args.backward_start
    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(schedule, schedule_phase='test')
    diffusion.feed_data(fixed_batch)

    net = diffusion.netG.module if isinstance(diffusion.netG, nn.DataParallel) else diffusion.netG
    stem = net.denoise_fn.downs[0]
    sl = cond_slice(args.target, opt['dem'])
    w0 = stem.weight.data.clone()
    w_noisy_norm = w0[:, 0:4].norm().item()
    w_c_norm = w0[:, sl].norm().item()
    logger.info(f'Stem conv weight {tuple(w0.shape)}; scaling input channels {sl.start}:{sl.stop} '
                f'(target={args.target}). ||W_noisy||={w_noisy_norm:.4f}, ||W_c||={w_c_norm:.4f}')
    logger.info(f"backward_start = {schedule.get('backward_start')}")

    meta = opt['datasets']['meta']
    norm_range = (meta['norm_min'], meta['norm_max'])
    max_depth = meta['max_depth']
    hr_depth = Metrics.unnormalize(fixed_batch['HR'], max_depth, min_max=norm_range).cpu()
    cg_depth = Metrics.unnormalize(fixed_batch['SR'], max_depth, min_max=norm_range).cpu()
    flood_mask = hr_depth > args.flood_threshold_cm
    cg_rmse_flood = torch.sqrt(torch.mean((cg_depth - hr_depth)[flood_mask] ** 2)).item()
    logger.info(f'Flooded pixels: {flood_mask.float().mean().item():.1%} of raster; '
                f'coarse-grid RMSE on them: {cg_rmse_flood:.3f} cm')

    rows = []
    start = time.time()
    for s in args.scales:
        stem.weight.data.copy_(w0)
        stem.weight.data[:, sl] *= s

        samples, per_run_rmse = [], []
        for k in range(args.n_samples):
            torch.manual_seed(args.seed + k)  # same K noise draws at every s
            diffusion.test()
            sr = diffusion.get_current_visuals()['SR'].clamp(norm_range[0], norm_range[1])
            sr_depth = Metrics.unnormalize(sr, max_depth, min_max=norm_range)
            samples.append(sr_depth)
            per_run_rmse.append(torch.sqrt(torch.mean((sr_depth - hr_depth)[flood_mask] ** 2)).item())
        ensemble = torch.stack(samples, dim=0)

        row = {
            'scale': s,
            'per_run_rmse_flooded_cm_mean': float(np.mean(per_run_rmse)),
            'per_run_rmse_flooded_cm_cv': float(np.std(per_run_rmse) / np.mean(per_run_rmse)),
            'all_pixels': coverage_and_stats(ensemble, hr_depth, args.ci),
            'flooded_pixels': coverage_and_stats(ensemble, hr_depth, args.ci, mask=flood_mask),
        }
        rows.append(row)
        f = row['flooded_pixels']
        logger.info(f"s={s:5.2f} | flooded: cov={f['coverage']:.3f} spread={f['spread_cm']:.3f}cm "
                    f"ens-mean RMSE={f['ensemble_mean_rmse_cm']:.3f}cm ss={f['spread_skill_ratio']:.3f} "
                    f"PIw={f['mean_pi_width_cm']:.3f}cm | per-run RMSE CV={row['per_run_rmse_flooded_cm_cv']:.4f} "
                    f"(elapsed {time.time() - start:.0f}s)")
        del ensemble, samples

    stem.weight.data.copy_(w0)

    # Ceiling read-out: how far s moved flooded spread relative to the unscaled checkpoint
    base = next((r for r in rows if r['scale'] == 1.0), None)
    if base is not None:
        b = base['flooded_pixels']
        best = max(rows, key=lambda r: r['flooded_pixels']['spread_skill_ratio'])
        logger.info('=' * 60)
        logger.info(f"s=1 baseline (flooded): spread={b['spread_cm']:.3f}cm, ss={b['spread_skill_ratio']:.3f}, "
                    f"cov={b['coverage']:.3f}, RMSE={b['ensemble_mean_rmse_cm']:.3f}cm")
        bf = best['flooded_pixels']
        logger.info(f"Best spread/skill at s={best['scale']:g}: ss={bf['spread_skill_ratio']:.3f} "
                    f"(x{bf['spread_cm'] / b['spread_cm']:.2f} spread), cov={bf['coverage']:.3f}, "
                    f"RMSE={bf['ensemble_mean_rmse_cm']:.3f}cm "
                    f"({bf['ensemble_mean_rmse_cm'] / b['ensemble_mean_rmse_cm'] - 1:+.1%} vs s=1)")
        logger.info('=' * 60)

    results = {
        'config': args.config,
        'target': args.target,
        'channel_slice': [sl.start, sl.stop],
        'backward_start': schedule.get('backward_start'),
        'n_scenes': n_scenes,
        'n_samples': args.n_samples,
        'ci': args.ci,
        'flood_threshold_cm': args.flood_threshold_cm,
        'w_noisy_norm': w_noisy_norm,
        'w_c_norm': w_c_norm,
        'coarse_grid_rmse_flooded_cm': cg_rmse_flood,
        'sweep': rows,
    }
    os.makedirs(opt['path']['results'], exist_ok=True)
    out_json = os.path.join(opt['path']['results'], 'cond_scale_probe.json')
    with open(out_json, 'w') as fh:
        json.dump(results, fh, indent=2)
    logger.info(f'Saved results to {out_json}')

    out_png = os.path.join(opt['path']['results'], 'cond_scale_probe.png')
    if plot_sweep(rows, out_png, args.ci):
        logger.info(f'Saved plot to {out_png}')
    else:
        logger.info('matplotlib not available; skipped plot (JSON has everything)')


if __name__ == '__main__':
    main()
