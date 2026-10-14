"""
Trained stem adapter (Variant 2 on the stem): a per-channel gain and shift on the U-Net stem
conv output (downs[0], 64 channels), trained with a CRPS loss through the full reverse chain.

  stochastic (default)  gamma = mu + sigma * eps, eps ~ N(0, I) redrawn for every ensemble
                        member; mu and log sigma learned per channel, plus a learned
                        deterministic shift beta. 192 params. This is the trained version of
                        stem_gain_probe.py: init mu=1, sigma=--init_sigma, beta=0 reproduces the
                        probe at that sigma (same eps draws at eval).
  deterministic         gamma = mu, beta learned. 128 params. Spread comes from reverse
                        diffusion only (the plan's literal Variant 2, a single adapter).

Loss: fair (unbiased) ensemble CRPS of M draws per scene on the decoded depth, in cm, clamped
to [norm_min, norm_max] first as eval does. Mean over flooded pixels plus --dry_weight times
the mean over dry pixels. Gradients run through every reverse step (per-step gradient
checkpointing on the U-Net) and through the frozen VAE decoder. The U-Net, both VAEs and
the encoder noise get no gradients; only the adapter trains.

Eval: the stem_gain_probe protocol (same eval/calib scenes, seeds, gain draws, scoring), on
the raw ensemble and with cp_cqr / cp_scale as post-hoc references. Targets from
stem_gain_probe v1/v2-wide (FloodLDM_Codebase_Context.md §11): beat CRPS 6.604cm and raw
IS 57.96cm, reach ~0.9 raw coverage at spread/skill near 1, ensemble-mean RMSE close to 11.91cm.

Training runs in bf16 autocast (fp16 risks gradient underflow over 50 chained steps). Eval
uses the shipped fp16 autocast, so with an identity adapter it reproduces `base`.

Usage (from the repo root):
    python scripts/train_stem_adapter.py -c config/wollombi-trnf.json -output_dir experiments/stem_adapter/v1
    # plumbing check:
    python scripts/train_stem_adapter.py -c config/wollombi-trnf.json -output_dir experiments/stem_adapter/smoke \
        --steps 10 --val_every 5 --n_val 2 --val_draws 2 --n_eval 4 --n_calib 4 --n_samples 6 --eval_init
"""
import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from decoder_uq import conformal, ensemble_metrics
from stem_gain_probe import scaled_conformal


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-output_dir', type=str, default='experiments/stem_adapter')
    parser.add_argument('--mode', choices=['stochastic', 'deterministic'], default='stochastic')
    parser.add_argument('--init_sigma', type=float, default=0.05,
                        help='Initial gain std (stochastic mode). 0.05 = best untrained CRPS in stem_gain_probe')
    # training
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch_scenes', type=int, default=2, help='Scenes per step')
    parser.add_argument('--draws', type=int, default=4, help='Ensemble members per scene per step (>= 2 for CRPS)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--dry_weight', type=float, default=0.1, help='Weight on the dry-pixel CRPS term')
    parser.add_argument('--n_train', type=int, default=2000, help='Train-split scenes in the pool')
    parser.add_argument('--log_every', type=int, default=25)
    # validation during training (held-out test scenes, disjoint from eval and calib)
    parser.add_argument('--val_every', type=int, default=200)
    parser.add_argument('--n_val', type=int, default=8)
    parser.add_argument('--val_draws', type=int, default=8)
    # final eval (stem_gain_probe protocol)
    parser.add_argument('--n_eval', type=int, default=40)
    parser.add_argument('--n_calib', type=int, default=40)
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--eval_init', action='store_true',
                        help='Also evaluate the untrained init (stochastic: should reproduce the probe at init_sigma)')
    parser.add_argument('--ci', type=float, default=0.90)
    parser.add_argument('--flood_threshold_cm', type=float, default=5.0)
    parser.add_argument('--cp_eps_cm', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-debug', '-d', action='store_true')
    args = parser.parse_args()
    args.phase = 'test'
    if args.draws < 2:
        parser.error('--draws must be >= 2: the CRPS spread term needs at least two members')
    return args


# ------------------------------------------------------------------ adapter

class StemAdapter(nn.Module):
    """out * gamma + beta on the stem conv output, through a forward hook. gamma/beta for the
    current forward pass are set per batch element with set_draw(); None = identity."""

    def __init__(self, stem, mode, init_sigma):
        super().__init__()
        C = stem.out_channels
        self.mode = mode
        self.mu = nn.Parameter(torch.ones(C))
        self.beta = nn.Parameter(torch.zeros(C))
        if mode == 'stochastic':
            self.log_sigma = nn.Parameter(torch.full((C,), math.log(init_sigma)))
        self.current = None
        self.handle = stem.register_forward_hook(self._hook)

    def _hook(self, module, inputs, out):
        if self.current is None:
            return None
        g, b = self.current
        return out * g.to(out.dtype)[:, :, None, None] + b.to(out.dtype)[None, :, None, None]

    def sigma(self):
        return self.log_sigma.exp() if self.mode == 'stochastic' else torch.zeros_like(self.mu)

    def gains(self, eps):
        """(N, C) gains for N draws from (N, C) standard-normal eps (ignored if deterministic)."""
        if self.mode == 'stochastic':
            return self.mu + self.sigma() * eps
        return self.mu.expand(eps.shape[0], -1)

    def summary(self):
        s = self.sigma().detach()
        return {'mu_mean': self.mu.mean().item(), 'mu_std': self.mu.std().item(),
                'sigma_mean': s.mean().item(), 'sigma_min': s.min().item(), 'sigma_max': s.max().item(),
                'beta_abs_mean': self.beta.abs().mean().item()}


def fair_crps(ens, y):
    """Per-pixel fair CRPS of (M, ...) members vs (...) truth:
    E|X - y| - sum_{i,j}|X_i - X_j| / (2 M (M - 1)), the pair sum taken from sorted members."""
    M = ens.shape[0]
    term1 = (ens - y).abs().mean(0)
    w = (2 * torch.arange(1, M + 1, device=ens.device, dtype=ens.dtype) - M - 1).view(-1, *[1] * y.dim())
    pair = (w * ens.sort(0).values).sum(0) / (M * (M - 1))
    return term1 - pair


# ------------------------------------------------------------------ main

def main():
    import core.logger as Logger
    import data as Data
    import model as Model

    args = parse_args()
    opt = Logger.dict_to_nonedict(Logger.parse(args))
    logger = Logger.setup_logger('base', opt['path']['log'], 'train_stem_adapter', screen=True)
    logger.info(Logger.dict2str(opt))
    logger.info(f'train_stem_adapter args: {vars(args)}')
    torch.backends.cudnn.benchmark = True
    if opt['model']['which_model_G'] != 'ddpm':
        raise SystemExit('train_stem_adapter needs a latent (ddpm) config')

    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['test'], schedule_phase='test')
    net = diffusion.netG.module if isinstance(diffusion.netG, nn.DataParallel) else diffusion.netG
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    dev = diffusion.device
    T = net.num_timesteps
    bs = opt['model']['beta_schedule']['test'].get('backward_start')
    bs = None if not bs or int(bs) >= T else int(bs)
    if bs is None:
        logger.info('warning: full 1000-step chain; training backprops through every step and will be slow')

    meta = opt['datasets']['meta']
    norm_range = (meta['norm_min'], meta['norm_max'])
    cm_per_unit = meta['max_depth'] / (norm_range[1] - norm_range[0])
    to_cm = lambda x: (x.float().clamp(*norm_range) - norm_range[0]) * cm_per_unit  # = Metrics.unnormalize on clamped
    amp_eval = bool(opt['amp']) and dev.type == 'cuda'

    adapter = StemAdapter(net.denoise_fn.downs[0], args.mode, args.init_sigma).to(dev)
    n_params = sum(p.numel() for p in adapter.parameters())
    logger.info(f'stem adapter ({args.mode}): {n_params} params on {net.denoise_fn.downs[0].out_channels} channels')

    # ---------------------------------------------------------- reverse chain
    def encode_cond(lr, dem):
        with torch.no_grad(), torch.autocast('cuda', enabled=amp_eval):
            lr_lat = net.vae.encode(lr)[0].float()
            dem_lat = net.vae_dem.encode(dem)[0].float() if net.dem else None
        return lr_lat, dem_lat

    def unet_step(x, lr_lat, dem_lat, t_tensor, g, b):
        adapter.current = (g, b)
        cond = [x, lr_lat, dem_lat] if net.dem else [x, lr_lat]
        out = net.denoise_fn(torch.cat(cond, dim=1), t_tensor)
        adapter.current = None
        return out

    def prev_x(x, eps_pred, t):
        # ddpm_modules.diffusion.sample_prev_timestep without its @torch.no_grad(); same arithmetic
        # and the same randn_like call, so the RNG stream matches the shipped sampler
        mean = (x - (1 - net.alphas[t]) * eps_pred / torch.sqrt(1 - net.alphas_cumprod[t])) / torch.sqrt(net.alphas[t])
        if t == 0:
            return mean
        var = (1 - net.alphas_cumprod[t - 1]) / (1 - net.alphas_cumprod[t]) * net.betas[t]
        return mean + var ** 0.5 * torch.randn_like(x)

    def rollout(lr_lat, dem_lat, g, b, grad):
        """Final latent for a batch. g: (N, C) gains or None (identity). With grad, every U-Net
        step is checkpointed, so memory is one step's activations plus the (N, 4, 64, 64) chain."""
        B = lr_lat.shape[0]
        if bs is None:
            steps, x = T, torch.randn_like(lr_lat)
        else:
            steps = bs
            x = net.add_noise(lr_lat, torch.randn_like(lr_lat), torch.full((B,), bs, dtype=torch.long, device=dev))
        for t in reversed(range(steps)):
            t_tensor = torch.full((B,), t, dtype=torch.long, device=dev)
            if g is None:
                adapter.current = None
                cond = [x, lr_lat, dem_lat] if net.dem else [x, lr_lat]
                eps_pred = net.denoise_fn(torch.cat(cond, dim=1), t_tensor)
            elif grad:
                eps_pred = checkpoint(unet_step, x, lr_lat, dem_lat, t_tensor, g, b, use_reentrant=False)
            else:
                eps_pred = unet_step(x, lr_lat, dem_lat, t_tensor, g, b)
            x = prev_x(x, eps_pred.float(), t)
        return x

    # ---------------------------------------------------------- data
    base_opt = dict(opt['datasets']['test'], data_len=-1, use_shuffle=False)
    test_set = Data.create_dataset(base_opt, 'test', meta, latent=False, dem=opt['dem'])
    train_set = Data.create_dataset(base_opt, 'train', meta, latent=False, dem=opt['dem'])
    # eval / calib exactly as decoder_uq.py and stem_gain_probe.py
    rng = np.random.default_rng(args.seed)
    eval_idx = list(range(min(args.n_eval, len(test_set))))
    rest = np.arange(len(eval_idx), len(test_set))
    calib_idx = sorted(rng.choice(rest, size=min(args.n_calib, len(rest)), replace=False).tolist())
    # validation: other test scenes, disjoint from both; training: random train-split scenes
    rng2 = np.random.default_rng(args.seed + 10)
    val_pool = np.setdiff1d(rest, calib_idx)
    val_idx = sorted(rng2.choice(val_pool, size=min(args.n_val, len(val_pool)), replace=False).tolist())
    train_idx = sorted(rng2.choice(len(train_set), size=min(args.n_train, len(train_set)), replace=False).tolist())

    def one_batch(ds, idx):
        return next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=len(idx),
                                                     shuffle=False, num_workers=2, pin_memory=True)))

    gen = torch.Generator().manual_seed(args.seed + 20)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_set, train_idx),
                                               batch_size=args.batch_scenes, shuffle=True, drop_last=True,
                                               num_workers=2, pin_memory=True, generator=gen)

    def batch_cond(batch):
        return batch['SR'].to(dev), batch['DEM'].to(dev) if 'DEM' in batch else None

    results = {'args': vars(args), 'config': args.config, 'backward_start': bs, 'n_params': n_params,
               'val_files': None, 'train_log': [], 'val_log': [], 'eval': {}}
    out_dir = opt['path']['results']
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'train_stem_adapter.json')
    ckpt_path = os.path.join(out_dir, 'stem_adapter.pt')

    def dump():
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    def save():
        torch.save({'state_dict': adapter.state_dict(), 'mode': args.mode, 'args': vars(args)}, ckpt_path)

    # ---------------------------------------------------------- validation (cheap, no_grad)
    val_batch = one_batch(test_set, val_idx)
    results['val_files'] = list(val_batch['filename'])
    val_y = to_cm(val_batch['HR'])
    val_mask = val_y > args.flood_threshold_cm
    val_eps = torch.randn(args.val_draws, adapter.mu.numel(), generator=torch.Generator().manual_seed(args.seed + 30))

    @torch.no_grad()
    def validate():
        lr, dem = batch_cond(val_batch)
        torch.manual_seed(args.seed + 31)
        draws = []
        for k in range(args.val_draws):
            lr_lat, dem_lat = encode_cond(lr, dem)
            g = adapter.gains(val_eps[k:k + 1].to(dev)).expand(lr.shape[0], -1)
            with torch.autocast('cuda', enabled=amp_eval):
                x = rollout(lr_lat, dem_lat, g, adapter.beta, grad=False)
                draws.append(to_cm(net.vae.decode(x)).cpu()[val_mask])
        return ensemble_metrics(torch.stack(draws, 0), val_y[val_mask], args.ci)

    # ---------------------------------------------------------- final eval (stem_gain_probe protocol)
    def evaluate(tag):
        t0 = time.time()
        ev_b, ca_b = one_batch(test_set, eval_idx), one_batch(test_set, calib_idx)
        ev_y, ca_y = to_cm(ev_b['HR']), to_cm(ca_b['HR'])
        ev_m, ca_m = ev_y > args.flood_threshold_cm, ca_y > args.flood_threshold_cm
        # same gain draws as stem_gain_probe (one eps vector per draw, seed + 3)
        eps = torch.randn(args.n_samples, adapter.mu.numel(), generator=torch.Generator().manual_seed(args.seed + 3))

        @torch.no_grad()
        def ensemble(batch, mask, seed):
            lr, dem = batch_cond(batch)
            torch.manual_seed(seed)  # seeded once per set, as decoder_uq / stem_gain_probe
            out = []
            for k in range(args.n_samples):
                lr_lat, dem_lat = encode_cond(lr, dem)
                g = adapter.gains(eps[k:k + 1].to(dev)).expand(lr.shape[0], -1)
                with torch.autocast('cuda', enabled=amp_eval):
                    x = rollout(lr_lat, dem_lat, g, adapter.beta, grad=False)
                    out.append(to_cm(net.vae.decode(x)).cpu()[mask])
            return torch.stack(out, 0)

        E, C = ensemble(ev_b, ev_m, args.seed), ensemble(ca_b, ca_m, args.seed + 2)
        y_ev, y_ca = ev_y[ev_m], ca_y[ca_m]
        draw_rmse = torch.sqrt(((E - y_ev) ** 2).mean(1))
        r = {'raw': ensemble_metrics(E, y_ev, args.ci),
             'cp_cqr': conformal(C, y_ca, E, y_ev, args.ci, 'cqr', args.cp_eps_cm),
             'cp_scale': scaled_conformal(C, y_ca, E, y_ev, args.ci, args.cp_eps_cm),
             'per_draw_rmse_cm': {'mean': draw_rmse.mean().item(), 'min': draw_rmse.min().item(),
                                  'max': draw_rmse.max().item()},
             'adapter': adapter.summary(), 'seconds': time.time() - t0}
        results['eval'][tag] = r
        dump()
        logger.info(f'eval [{tag}] ({r["seconds"]:.0f}s); per-draw flooded RMSE {r["per_draw_rmse_cm"]["mean"]:.3f}cm '
                    f'(range {r["per_draw_rmse_cm"]["min"]:.3f}-{r["per_draw_rmse_cm"]["max"]:.3f})')
        logger.info(f"{'method':>9} {'cov':>6} {'width':>7} {'IS':>7} {'CRPS':>6} {'RMSE':>7} {'ss':>6}")
        for name in ('raw', 'cp_cqr', 'cp_scale'):
            m = r[name]
            crps = f"{m['crps_cm']:6.3f}" if 'crps_cm' in m else f"{'-':>6}"
            ss = f"{m['spread_skill_ratio']:6.3f}" if 'spread_skill_ratio' in m else f"{'-':>6}"
            logger.info(f"{name:>9} {m['coverage']:6.3f} {m['mean_pi_width_cm']:7.3f} "
                        f"{m['interval_score_cm']:7.3f} {crps} {m['ensemble_mean_rmse_cm']:7.3f} {ss}")

    # Identity check: with gamma=1, beta=0 the hook must not change the U-Net output
    with torch.no_grad(), torch.autocast('cuda', enabled=amp_eval):
        lr, dem = batch_cond(val_batch)
        lr_lat, dem_lat = encode_cond(lr[:2], dem[:2] if dem is not None else None)
        x = torch.randn_like(lr_lat)
        t_tensor = torch.full((x.shape[0],), 10, dtype=torch.long, device=dev)
        cond = [x, lr_lat, dem_lat] if net.dem else [x, lr_lat]
        ref = net.denoise_fn(torch.cat(cond, dim=1), t_tensor)
        ones = torch.ones(x.shape[0], adapter.mu.numel(), device=dev)
        diff = (ref - unet_step(x, lr_lat, dem_lat, t_tensor, ones, torch.zeros_like(adapter.beta))).abs().max().item()
    logger.info(f'identity-adapter check: max |diff| {diff:.2e}')
    if diff > 1e-3:
        raise SystemExit('identity adapter changes the U-Net output; hook is wrong')

    if args.eval_init:
        evaluate('init')
    v = validate()
    results['val_log'].append({'step': 0, **v, 'adapter': adapter.summary()})
    logger.info(f"val step 0: flooded CRPS {v['crps_cm']:.3f} cov {v['coverage']:.3f} "
                f"ss {v['spread_skill_ratio']:.3f} RMSE {v['ensemble_mean_rmse_cm']:.3f}")
    dump()

    # ---------------------------------------------------------- train
    optim = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    M, t0, losses, step = args.draws, time.time(), [], 0
    train_eps_gen = torch.Generator().manual_seed(args.seed + 40)
    torch.manual_seed(args.seed + 41)
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            lr, dem = batch_cond(batch)
            B = lr.shape[0]
            # B scenes x M members; each member has its own encoder noise, chain noise and gain
            lr_r = lr.repeat_interleave(M, 0)
            dem_r = dem.repeat_interleave(M, 0) if dem is not None else None
            lr_lat, dem_lat = encode_cond(lr_r, dem_r)
            eps = torch.randn(B * M, adapter.mu.numel(), generator=train_eps_gen).to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                x = rollout(lr_lat, dem_lat, adapter.gains(eps), adapter.beta, grad=True)
                pred = to_cm(net.vae.decode(x)).squeeze(1)  # (B*M, H, W)
            y = to_cm(batch['HR'].to(dev)).squeeze(1)       # (B, H, W)
            crps = fair_crps(pred.view(B, M, *pred.shape[1:]).transpose(0, 1), y)
            fl = y > args.flood_threshold_cm
            loss_fl = crps[fl].mean() if fl.any() else crps.new_zeros(())
            loss = loss_fl + args.dry_weight * crps[~fl].mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip).item()
            optim.step()
            step += 1
            losses.append(loss_fl.item() if fl.any() else float('nan'))  # all-dry batch: no flooded CRPS
            if step == 1 or step % args.log_every == 0:
                s = adapter.summary()
                recent = float(np.nanmean(losses[-args.log_every:])) if not np.all(np.isnan(losses[-args.log_every:])) \
                    else float('nan')
                results['train_log'].append({'step': step, 'flooded_crps_cm': recent, 'grad_norm': gnorm, **s})
                logger.info(f'step {step}/{args.steps} flooded CRPS {recent:.3f}cm '
                            f'|g| {gnorm:.2e} sigma {s["sigma_mean"]:.4f} [{s["sigma_min"]:.4f}, {s["sigma_max"]:.4f}] '
                            f'mu {s["mu_mean"]:.4f}+-{s["mu_std"]:.4f} |beta| {s["beta_abs_mean"]:.2e} '
                            f'({time.time() - t0:.0f}s)')
            if step % args.val_every == 0 or step == args.steps:
                v = validate()
                results['val_log'].append({'step': step, **v, 'adapter': adapter.summary()})
                logger.info(f"val step {step}: flooded CRPS {v['crps_cm']:.3f} cov {v['coverage']:.3f} "
                            f"ss {v['spread_skill_ratio']:.3f} RMSE {v['ensemble_mean_rmse_cm']:.3f}")
                save()
                dump()

    save()
    logger.info(f'training done ({time.time() - t0:.0f}s); saved {ckpt_path}')
    evaluate('trained')
    logger.info('targets (stem_gain_probe, untrained): CRPS 6.604 (sigma=0.05), raw IS 57.96 / cov 0.858 '
                '(sigma=0.08); base RMSE 11.911; post-hoc reference cp_cqr IS 51.45')
    dump()
    logger.info(f'Saved {out_path}')


if __name__ == '__main__':
    main()
