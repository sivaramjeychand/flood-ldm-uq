"""
Fixed-input stochastic-ensemble coverage diagnostic.

Reproduces the method from Dr Viraj's EGU 2026 abstract (FYP_Context.md
S1.2 / S8.4 item 3): fix a small set of test scenes, draw K independent
stochastic reverse-diffusion samples per scene (same conditioning input,
different noise each time), and check whether the resulting empirical
prediction interval actually covers the fine-grid ground truth at the
nominal rate. The abstract found ~70% empirical coverage of a nominal 90%
interval, regardless of ensemble size -- this script checks whether the
same gap shows up on our own checkpoint/catchment.

Usage (mirrors sr.py's CLI):
    python scripts/coverage_diagnostic.py -c config/wollombi-trnf.json \
        -output_dir experiments/coverage_wollombi --n_scenes 10 --n_samples 50
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import core.logger as Logger
import core.metrics as Metrics
import data as Data
import model as Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                         help='JSON config file (same format as sr.py); must have a "test" dataset block')
    parser.add_argument('-output_dir', type=str, default='experiments/coverage_diagnostic')
    parser.add_argument('--n_scenes', type=int, default=10,
                         help='Number of fixed test scenes to evaluate')
    parser.add_argument('--n_samples', type=int, default=50,
                         help='Stochastic reverse-diffusion draws per scene (paper used 20-100 runs)')
    parser.add_argument('--ci', type=float, default=0.90,
                         help='Nominal central prediction interval to check coverage for, e.g. 0.90 for a 90%% PI')
    parser.add_argument('--convergence_checkpoints', type=int, nargs='*',
                         default=[5, 10, 20, 30, 40, 50],
                         help='Ensemble sizes to also report coverage/RMSE at, to check convergence')
    parser.add_argument('-debug', '-d', action='store_true')
    args = parser.parse_args()
    args.phase = 'test'  # required by core.logger.Logger.parse
    return args


def coverage_and_stats(ensemble, hr_depth, ci):
    """ensemble: (K, N, C, H, W) depth-space tensor. hr_depth: (N, C, H, W)."""
    alpha = 1 - ci
    lower = torch.quantile(ensemble, alpha / 2, dim=0)
    upper = torch.quantile(ensemble, 1 - alpha / 2, dim=0)
    inside = (hr_depth >= lower) & (hr_depth <= upper)
    ens_mean = ensemble.mean(dim=0)
    ens_mean_rmse = torch.sqrt(torch.mean((ens_mean - hr_depth) ** 2)).item()

    # Spread-skill: RMS of per-pixel ensemble std ("spread") vs. ensemble-mean RMSE ("skill").
    # A well-calibrated ensemble has spread ~= skill (ratio ~1). Ratio << 1 means the ensemble
    # is under-dispersed relative to its actual error -- the diversity-collapse signature from
    # the EGU abstract (deterministic conditioning suppressing sample diversity), as distinct
    # from under-coverage caused by a biased ensemble mean.
    pixel_std = ensemble.std(dim=0, unbiased=(ensemble.shape[0] > 1))
    spread = torch.sqrt(torch.mean(pixel_std ** 2)).item()
    mean_pi_width = (upper - lower).mean().item()

    return {
        'n_samples': int(ensemble.shape[0]),
        'coverage': inside.float().mean().item(),
        'ensemble_mean_rmse_cm': ens_mean_rmse,
        'spread_cm': spread,
        'spread_skill_ratio': spread / ens_mean_rmse if ens_mean_rmse > 0 else float('nan'),
        'mean_pi_width_cm': mean_pi_width,
    }


def main():
    args = parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    logger = Logger.setup_logger('base', opt['path']['log'], 'coverage_diagnostic', screen=True)
    logger.info(Logger.dict2str(opt))
    logger.info(f'Coverage diagnostic: n_scenes={args.n_scenes}, n_samples={args.n_samples}, ci={args.ci}')

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Fixed scene set: force a small, single-batch, unshuffled test set so the
    # same conditioning inputs are reused across all K stochastic draws.
    test_opt = dict(opt['datasets']['test'])
    test_opt['data_len'] = args.n_scenes
    test_opt['batch_size'] = args.n_scenes
    test_opt['use_shuffle'] = False

    test_set = Data.create_dataset(test_opt, 'test', opt['datasets']['meta'], latent=False, dem=opt['dem'])
    if len(test_set) < args.n_scenes:
        logger.warning(f"Only {len(test_set)} scenes available in {test_opt['dataroot']}, "
                        f'requested {args.n_scenes}. Using {len(test_set)}.')
    test_loader = Data.create_dataloader(test_set, test_opt, 'test')
    fixed_batch = next(iter(test_loader))
    n_scenes = fixed_batch['HR'].shape[0]

    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['test'], schedule_phase='test')
    diffusion.feed_data(fixed_batch)

    meta = opt['datasets']['meta']
    norm_range = (meta['norm_min'], meta['norm_max'])
    max_depth = meta['max_depth']

    hr_depth = Metrics.unnormalize(fixed_batch['HR'], max_depth, min_max=norm_range)
    cg_depth = Metrics.unnormalize(fixed_batch['SR'], max_depth, min_max=norm_range)
    cg_rmse = torch.sqrt(torch.mean((cg_depth - hr_depth) ** 2)).item()

    logger.info(f'Running {args.n_samples} stochastic reverse-diffusion passes over {n_scenes} fixed scenes...')
    samples = []
    per_run_rmse = []
    start = time.time()
    for k in range(args.n_samples):
        diffusion.test()  # feed_data was set once above; this call alone resamples the noise
        visuals = diffusion.get_current_visuals()
        sr = visuals['SR'].clamp(norm_range[0], norm_range[1])
        sr_depth = Metrics.unnormalize(sr, max_depth, min_max=norm_range)
        samples.append(sr_depth)

        run_rmse = torch.sqrt(torch.mean((sr_depth - hr_depth) ** 2)).item()
        per_run_rmse.append(run_rmse)
        logger.info(f'  sample {k + 1}/{args.n_samples}: RMSE={run_rmse:.3f} cm '
                    f'(elapsed {time.time() - start:.1f}s)')

    ensemble = torch.stack(samples, dim=0)  # (K, N, C, H, W)

    results = {
        'catchment': test_opt['catchment'],
        'n_scenes': n_scenes,
        'ci': args.ci,
        'coarse_grid_baseline_rmse_cm': cg_rmse,
        'per_run_rmse_cm': per_run_rmse,
    }
    checkpoints = sorted(set(cp for cp in args.convergence_checkpoints if cp <= args.n_samples) | {args.n_samples})
    results['convergence'] = [coverage_and_stats(ensemble[:cp], hr_depth, args.ci) for cp in checkpoints]

    logger.info('=' * 60)
    logger.info(f'Coarse-grid baseline RMSE: {cg_rmse:.3f} cm')
    logger.info(f'Individual-run RMSE: mean={np.mean(per_run_rmse):.3f}, '
                f'std={np.std(per_run_rmse):.3f} cm (EGU abstract found ~19-24cm)')
    for c in results['convergence']:
        logger.info(f"  K={c['n_samples']:3d}: empirical {args.ci:.0%} PI coverage = {c['coverage']:.3f}, "
                    f"ensemble-mean RMSE = {c['ensemble_mean_rmse_cm']:.3f} cm, "
                    f"spread = {c['spread_cm']:.3f} cm, spread/skill = {c['spread_skill_ratio']:.3f}, "
                    f"mean PI width = {c['mean_pi_width_cm']:.3f} cm")
    final = results['convergence'][-1]
    logger.info(f"Nominal coverage target: {args.ci:.2f}. "
                f"Under-coverage gap at K={final['n_samples']}: {args.ci - final['coverage']:.3f} "
                f"(EGU abstract found ~{args.ci - 0.70:.2f} gap, i.e. ~70% actual coverage)")
    if final['spread_skill_ratio'] < 0.9:
        logger.info(f"DIVERSITY COLLAPSE INDICATED: spread/skill ratio = {final['spread_skill_ratio']:.3f} << 1 "
                    f"-- ensemble spread ({final['spread_cm']:.3f} cm) under-represents actual error "
                    f"({final['ensemble_mean_rmse_cm']:.3f} cm), consistent with the EGU abstract's "
                    f"conditioning-driven diversity collapse, not a biased ensemble mean.")
    else:
        logger.info(f"No strong diversity collapse signal: spread/skill ratio = {final['spread_skill_ratio']:.3f} "
                    f"(near 1 = well-calibrated spread).")
    logger.info('=' * 60)

    out_path = os.path.join(opt['path']['results'], 'coverage_diagnostic.json')
    os.makedirs(opt['path']['results'], exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Saved full results to {out_path}')


if __name__ == '__main__':
    main()
