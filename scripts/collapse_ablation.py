"""
Is the diversity collapse caused by the latent space, or by the diffusion setup itself?

Runs three checks on one fixed batch of test scenes. The scenes, draw count, CI and
flood mask match scripts/coverage_diagnostic.py, so the numbers can be compared directly.

  1. VAE floor  (latent configs only, cheap, runs first)
       Encode the fine-grid ground truth, decode it again, and measure the error on
       flooded pixels. Every ensemble member goes through the same deterministic
       decoder, so no amount of latent diversity can cover this error. Decoding is
       done from the posterior mean and from K sampled encodes, and the sampled
       reconstructions are also scored as an "oracle ensemble".
  2. backward_start sweep
       Re-runs the K-draw ensemble for each start timestep, including a pure-noise
       start ('full'). Coverage and spread/skill are reported on all pixels and on
       flooded pixels only.
  3. Latent vs decoded spread  (latent configs only, computed inside the sweep)
       For every sweep setting, spread/skill is computed in latent space against the
       encoder mean of the ground truth, and in pixel space after decoding. If the
       latent ratio is healthy and the pixel ratio is not, the decoder is the part
       that squashes the spread.

The sweep also accepts a pixel-space (sr3) config. Run it with the same scene count,
draw count, threshold and backward_starts to get the matched DM-vs-LDM comparison.

Usage (from the repo root):
    python scripts/collapse_ablation.py -c config/wollombi-trnf.json \
        -output_dir experiments/collapse_wollombi --n_scenes 10 --n_samples 50 \
        --backward_starts 50 100 250 500 full
    # Just the VAE floor, for a quick first answer:
    python scripts/collapse_ablation.py -c config/wollombi-trnf.json \
        -output_dir experiments/collapse_wollombi --skip_sweep
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
import torch.nn.functional as F

import core.logger as Logger
import core.metrics as Metrics
import data as Data
import model as Model
from coverage_diagnostic import coverage_and_stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-output_dir', type=str, default='experiments/collapse_ablation')
    parser.add_argument('--n_scenes', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Draws per scene for each backward_start setting')
    parser.add_argument('--ci', type=float, default=0.90)
    parser.add_argument('--flood_threshold_cm', type=float, default=5.0)
    parser.add_argument('--backward_starts', type=str, nargs='+', default=['50', '100', '250', '500', 'full'],
                        help="Start timesteps to sweep. 'full' (or any value >= n_timestep) starts from pure noise "
                             "and runs all n_timestep reverse steps.")
    parser.add_argument('--cond_encode', choices=['sample', 'mean'], default='sample',
                        help="Latent path only. 'sample' re-samples the coarse-map/DEM latents on every draw, as the "
                             "shipped pipeline does. 'mean' uses the encoder mean, which removes VAE-encoder noise "
                             "from the ensemble.")
    parser.add_argument('--vae_floor_draws', type=int, default=20,
                        help='Number of sampled encode->decode reconstructions of the ground truth')
    parser.add_argument('--skip_vae_floor', action='store_true')
    parser.add_argument('--skip_sweep', action='store_true')
    parser.add_argument('--save_latents', action='store_true',
                        help='Save the (K, N, 4, 64, 64) final-latent stack for each setting (~33MB at K=50, N=10)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-debug', '-d', action='store_true')
    args = parser.parse_args()
    args.phase = 'test'
    return args


def _err_stats(pred, target, mask):
    """RMSE / MAE / bias / p95 |error| over mask==True entries."""
    err = (pred - target)[mask]
    if err.numel() == 0:
        return {k: float('nan') for k in ('rmse_cm', 'mae_cm', 'bias_cm', 'p95_abs_err_cm')}
    return {
        'rmse_cm': torch.sqrt(torch.mean(err ** 2)).item(),
        'mae_cm': err.abs().mean().item(),
        'bias_cm': err.mean().item(),
        'p95_abs_err_cm': float(np.percentile(err.abs().numpy(), 95)),
    }


def _spread_skill(ensemble, target, mask):
    """Unit-free spread/skill ratio. It works the same way in latent space and in pixel space."""
    std = ensemble.std(dim=0, unbiased=ensemble.shape[0] > 1)
    spread = torch.sqrt(torch.mean(std[mask] ** 2)).item()
    skill = torch.sqrt(torch.mean((ensemble.mean(dim=0) - target)[mask] ** 2)).item()
    return {'spread': spread, 'skill': skill, 'spread_skill_ratio': spread / skill if skill > 0 else float('nan')}


class Runner:
    def __init__(self, diffusion, opt, device):
        self.net = diffusion.netG.module if isinstance(diffusion.netG, nn.DataParallel) else diffusion.netG
        self.net.eval()
        self.latent = opt['model']['which_model_G'] == 'ddpm'
        self.amp = bool(opt['amp']) and device.type == 'cuda'
        self.device = device
        self.T = self.net.num_timesteps

    def _autocast(self):
        return torch.autocast('cuda', enabled=self.amp)

    def resolve_bs(self, bs):
        return None if bs == 'full' or int(bs) >= self.T else int(bs)

    @torch.no_grad()
    def sample(self, lr, dem, backward_start, cond_encode):
        """One reverse-diffusion draw. Returns (decoded pixel output, final latent or None)."""
        with self._autocast():
            if not self.latent:
                self.net.backward_start = backward_start
                return self.net.super_resolution(lr, dem).float(), None

            # Same computation as ddpm_modules.GaussianDiffusion.super_resolution, but with a
            # selectable encoder mode and the final latent returned before decoding.
            z_s, z_mu, _ = self.net.vae.encode(lr)
            lr_lat = z_s if cond_encode == 'sample' else z_mu
            if self.net.dem:
                d_s, d_mu, _ = self.net.vae_dem.encode(dem)
                dem_lat = d_s if cond_encode == 'sample' else d_mu
            B = lr_lat.shape[0]
            if backward_start is None:
                steps = self.T
                x = torch.randn_like(lr_lat)
            else:
                steps = backward_start
                t = torch.full((B,), backward_start, dtype=torch.long, device=lr_lat.device)
                x = self.net.add_noise(lr_lat, torch.randn_like(lr_lat), t)
            for t in reversed(range(steps)):
                t_tensor = torch.full((B,), t, dtype=torch.long, device=lr_lat.device)
                cond = [x, lr_lat, dem_lat] if self.net.dem else [x, lr_lat]
                noise_pred = self.net.denoise_fn(torch.cat(cond, dim=1), t_tensor)
                x = self.net.sample_prev_timestep(x, noise_pred, t)
            return self.net.vae.decode(x).float(), x.float()

    @torch.no_grad()
    def encode(self, img):
        with self._autocast():
            z, mu, logvar = self.net.vae.encode(img)
        return z.float(), mu.float(), logvar.float()

    @torch.no_grad()
    def decode(self, z):
        with self._autocast():
            return self.net.vae.decode(z).float()


def main():
    args = parse_args()
    opt = Logger.dict_to_nonedict(Logger.parse(args))
    logger = Logger.setup_logger('base', opt['path']['log'], 'collapse_ablation', screen=True)
    logger.info(Logger.dict2str(opt))
    torch.backends.cudnn.benchmark = True

    test_opt = dict(opt['datasets']['test'])
    test_opt.update(data_len=args.n_scenes, batch_size=args.n_scenes, use_shuffle=False)
    test_set = Data.create_dataset(test_opt, 'test', opt['datasets']['meta'], latent=False, dem=opt['dem'])
    batch = next(iter(Data.create_dataloader(test_set, test_opt, 'test')))
    n_scenes = batch['HR'].shape[0]

    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['test'], schedule_phase='test')
    runner = Runner(diffusion, opt, diffusion.device)
    dev = diffusion.device

    # Tensors are moved to the device explicitly. feed_data() is never called, which
    # avoids the in-place set_device mutation (FloodLDM_Codebase_Context.md §8 item 14).
    hr, lr = batch['HR'].to(dev), batch['SR'].to(dev)
    dem = batch['DEM'].to(dev) if 'DEM' in batch else None

    meta = opt['datasets']['meta']
    norm_range = (meta['norm_min'], meta['norm_max'])
    to_depth = lambda x: Metrics.unnormalize(x.float().cpu().clamp(*norm_range), meta['max_depth'], min_max=norm_range)
    hr_depth = to_depth(hr)
    cg_depth = to_depth(lr)
    flood_mask = hr_depth > args.flood_threshold_cm
    all_mask = torch.ones_like(flood_mask)

    results = {
        'config': args.config,
        'which_model_G': opt['model']['which_model_G'],
        'catchment': test_opt['catchment'],
        'filenames': list(batch['filename']),
        'n_scenes': n_scenes,
        'n_samples': args.n_samples,
        'ci': args.ci,
        'flood_threshold_cm': args.flood_threshold_cm,
        'flooded_pixel_fraction': flood_mask.float().mean().item(),
        'cond_encode': args.cond_encode if runner.latent else None,
        'coarse_grid_baseline': {'all': _err_stats(cg_depth, hr_depth, all_mask),
                                 'flooded': _err_stats(cg_depth, hr_depth, flood_mask)},
    }
    out_dir = opt['path']['results']
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'collapse_ablation.json')

    def dump():
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    logger.info(f"{n_scenes} scenes, flooded fraction {results['flooded_pixel_fraction']:.1%}, "
                f"coarse-grid flooded RMSE {results['coarse_grid_baseline']['flooded']['rmse_cm']:.3f} cm")

    # ---------------------------------------------------------------- 1. VAE floor
    hr_mu_lat = None
    if runner.latent:
        torch.manual_seed(args.seed)
        _, hr_mu_lat, hr_logvar = runner.encode(hr)
        if not args.skip_vae_floor:
            recon_mean = to_depth(runner.decode(hr_mu_lat))
            recon_samples = torch.stack(
                [to_depth(runner.decode(runner.encode(hr)[0])) for _ in range(args.vae_floor_draws)], dim=0)
            # 0.18215 * exp(0.5 * logvar) is the encoder std in the scaled latent units the UNet sees
            enc_sigma = 0.18215 * torch.exp(0.5 * hr_logvar)
            results['vae_floor'] = {
                'recon_from_mean': {'all': _err_stats(recon_mean, hr_depth, all_mask),
                                    'flooded': _err_stats(recon_mean, hr_depth, flood_mask)},
                'recon_from_single_sample': {'flooded': _err_stats(recon_samples[0], hr_depth, flood_mask)},
                'oracle_ensemble_all': coverage_and_stats(recon_samples, hr_depth, args.ci),
                'oracle_ensemble_flooded': coverage_and_stats(recon_samples, hr_depth, args.ci, mask=flood_mask),
                'hr_encoder_sigma_rms_latent': torch.sqrt(torch.mean(enc_sigma ** 2)).item(),
                'hr_latent_rms': torch.sqrt(torch.mean(hr_mu_lat ** 2)).item(),
            }
            vf = results['vae_floor']
            logger.info('=' * 60)
            logger.info('VAE FLOOR: ground truth -> encode -> decode')
            for name in ('all', 'flooded'):
                s = vf['recon_from_mean'][name]
                logger.info(f"  [{name:7s}] RMSE {s['rmse_cm']:.3f} cm, MAE {s['mae_cm']:.3f}, "
                            f"bias {s['bias_cm']:+.3f}, p95|err| {s['p95_abs_err_cm']:.3f}")
            o = vf['oracle_ensemble_flooded']
            logger.info(f"  Oracle ensemble ({args.vae_floor_draws} sampled encodes of the TRUTH), flooded: "
                        f"coverage {o['coverage']:.3f}, spread/skill {o['spread_skill_ratio']:.3f}")
            logger.info(f"  Encoder sigma (latent units) {vf['hr_encoder_sigma_rms_latent']:.4f} vs "
                        f"latent RMS {vf['hr_latent_rms']:.4f}")
            logger.info('  Compare the flooded RMSE above with the ensemble-mean flooded RMSE in the sweep below.')
            dump()
            del recon_samples

    # ---------------------------------------------------------------- 2+3. backward_start sweep
    if not args.skip_sweep:
        results['sweep'] = []
        lat_mask = None
        if runner.latent:
            # A latent cell counts as flooded if any pixel in its 8x8 footprint is flooded
            lat_mask = (F.max_pool2d(flood_mask.float(), kernel_size=8) > 0).expand(-1, 4, -1, -1)

        for bs_arg in args.backward_starts:
            bs = runner.resolve_bs(bs_arg)
            steps = runner.T if bs is None else bs
            label = 'full (pure noise)' if bs is None else str(bs)
            if runner.latent:
                abar = 0.0 if bs is None else runner.net.alphas_cumprod[bs].item()
            else:
                abar = 0.0 if bs is None else float(runner.net.sqrt_alphas_cumprod_prev[bs]) ** 2
            logger.info('=' * 60)
            logger.info(f'SWEEP backward_start={label}: {steps} steps, start signal fraction '
                        f'sqrt(abar)={abar ** 0.5:.4f}, noise std={(1 - abar) ** 0.5:.4f}')

            torch.manual_seed(args.seed)
            np.random.seed(args.seed)  # the sr3 path draws its start noise level with np.random
            pix, lats, per_run = [], [], []
            t0 = time.time()
            for k in range(args.n_samples):
                out, lat = runner.sample(lr, dem, bs, args.cond_encode)
                d = to_depth(out)
                pix.append(d)
                if lat is not None:
                    lats.append(lat.cpu())
                per_run.append(_err_stats(d, hr_depth, flood_mask)['rmse_cm'])
                if (k + 1) % 10 == 0 or k == 0:
                    logger.info(f'  draw {k + 1}/{args.n_samples} flooded RMSE {per_run[-1]:.3f} cm '
                                f'({time.time() - t0:.0f}s)')
            ens = torch.stack(pix, dim=0)

            row = {
                'backward_start': bs_arg, 'n_steps': steps, 'start_sqrt_abar': abar ** 0.5,
                'seconds': time.time() - t0,
                'per_run_flooded_rmse_mean_cm': float(np.mean(per_run)),
                'per_run_flooded_rmse_std_cm': float(np.std(per_run)),
                'pixel_all': coverage_and_stats(ens, hr_depth, args.ci),
                'pixel_flooded': coverage_and_stats(ens, hr_depth, args.ci, mask=flood_mask),
            }
            pf = row['pixel_flooded']
            logger.info(f"  PIXEL flooded: coverage {pf['coverage']:.3f}, ens-mean RMSE "
                        f"{pf['ensemble_mean_rmse_cm']:.3f} cm, spread {pf['spread_cm']:.3f} cm, "
                        f"spread/skill {pf['spread_skill_ratio']:.3f}, PI width {pf['mean_pi_width_cm']:.3f} cm")

            if runner.latent:
                lat_ens = torch.stack(lats, dim=0)
                la = _spread_skill(lat_ens, hr_mu_lat.cpu(), torch.ones_like(lat_mask))
                lf = _spread_skill(lat_ens, hr_mu_lat.cpu(), lat_mask)
                row['latent_all'], row['latent_flooded'] = la, lf
                # If the ratios match, the decoder is passing spread and error through equally.
                # If the pixel ratio is much lower, the decoder is squashing the spread.
                row['decoder_ratio_attenuation_flooded'] = (pf['spread_skill_ratio'] / lf['spread_skill_ratio']
                                                            if lf['spread_skill_ratio'] else float('nan'))
                logger.info(f"  LATENT flooded: spread {lf['spread']:.4f}, skill vs mu(HR) {lf['skill']:.4f}, "
                            f"spread/skill {lf['spread_skill_ratio']:.3f}  ->  pixel/latent ratio "
                            f"{row['decoder_ratio_attenuation_flooded']:.3f}")
                if args.save_latents:
                    torch.save(lat_ens.half(), os.path.join(out_dir, f'latents_bs{bs_arg}.pt'))
                del lat_ens
            results['sweep'].append(row)
            dump()
            del ens, pix

        logger.info('=' * 60)
        logger.info(f"{'bs':>6} {'steps':>5} {'cov_fl':>7} {'ss_fl':>6} {'rmse_fl':>8} {'lat_ss':>7} {'sec':>6}")
        for r in results['sweep']:
            lat_ss = r.get('latent_flooded', {}).get('spread_skill_ratio', float('nan'))
            logger.info(f"{r['backward_start']:>6} {r['n_steps']:>5} {r['pixel_flooded']['coverage']:7.3f} "
                        f"{r['pixel_flooded']['spread_skill_ratio']:6.3f} "
                        f"{r['pixel_flooded']['ensemble_mean_rmse_cm']:8.3f} {lat_ss:7.3f} {r['seconds']:6.0f}")
        if 'vae_floor' in results:
            logger.info(f"VAE floor (flooded RMSE): {results['vae_floor']['recon_from_mean']['flooded']['rmse_cm']:.3f} cm")

    dump()
    logger.info(f'Saved {out_path}')


if __name__ == '__main__':
    main()
