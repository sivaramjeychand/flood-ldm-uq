"""
Decoder-side UQ: a deep ensemble of SSF-adapted VAE decoders vs split-conformal calibration.

collapse_ablation.py put most of the flooded-pixel error in the VAE decoder (the "VAE floor",
FloodLDM_Codebase_Context.md §11), where nothing inside the diffusion UNet can reach it.
Both methods here act after the reverse chain, on the decoder side.

Shared setup
  eval  : the first --n_eval test scenes, the same scenes coverage_diagnostic.py and
          collapse_ablation.py use, so `base` reproduces their numbers.
  calib : --n_calib other test scenes, drawn at random from the rest of the test split.
  train : --n_train scenes drawn at random from the train split (SSF training only).
  One pool of LDM latents is generated per set with the shipped sampler (the config's
  backward_start, sampled encoder) and reused by every method. The methods therefore
  differ only in what happens after the chain.

Methods (every ensemble uses --n_samples decodes per scene, so budgets match)
  base             frozen decoder, one decode per latent.
  ssf_single       SSF member 0 decodes every latent. Shows what fine-tuning alone does.
  ssf_ens          Deep ensemble. Each of --n_members SSF adapters (per-channel scale and
                   shift after every Conv2d / Linear in the decoder, ~29k params) is trained
                   on its own: SSF's N(1, s) / N(0, s) random init, its own bootstrap of the
                   train pool, its own batch order. Latent i is decoded by member i mod M.
  cp_abs           Split conformal on the base ensemble mean, score |y - mean|. Fixed width.
  cp_norm          Score |y - mean| / (std + eps). Width follows the ensemble spread.
  cp_cqr           Conformalised quantiles, score max(lo - y, y - hi) on the ensemble's own PI.
  ssf_ens+cp_norm  cp_norm on top of the SSF ensemble.

Conformal scores use flooded calibration pixels (truth >= --flood_threshold_cm), which is
how coverage is reported. Pixels within a scene are correlated, so the finite-sample
guarantee only holds approximately; the eval coverage is the real check. Conformal
intervals are clipped at 0 cm (depth can't be negative). That only narrows them and
can't change coverage.

Usage (from the repo root):
    python scripts/decoder_uq.py -c config/wollombi-trnf.json -output_dir experiments/decoder_uq/v1
    # quick plumbing check:
    python scripts/decoder_uq.py -c config/wollombi-trnf.json -output_dir experiments/decoder_uq/smoke \
        --n_eval 4 --n_calib 4 --n_train 16 --n_samples 6 --n_members 2 --steps 20
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-output_dir', type=str, default='experiments/decoder_uq')
    parser.add_argument('--n_eval', type=int, default=40)
    parser.add_argument('--n_calib', type=int, default=40)
    parser.add_argument('--n_train', type=int, default=2000)
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Decodes per scene for every ensemble (latent draws per eval/calib scene)')
    parser.add_argument('--n_members', type=int, default=5)
    parser.add_argument('--steps', type=int, default=2000, help='Optimiser steps per SSF member')
    parser.add_argument('--batch_size', type=int, default=4, help='SSF training batch (decoder runs at 512x512)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--init_std', type=float, default=0.02,
                        help="SSF init: gamma ~ N(1, s), beta ~ N(0, s), as in the SSF paper")
    parser.add_argument('--flood_weight', type=float, default=1.0,
                        help='Loss weight on flooded pixels (1 = plain MSE, the deep-ensemble default)')
    parser.add_argument('--train_target', choices=['ldm', 'mu'], default='ldm',
                        help="'ldm': train on LDM-generated latents (calibrates the whole pipeline). "
                             "'mu': train on encoder means of the truth (fits the VAE floor only).")
    parser.add_argument('--no_bootstrap', action='store_true')
    parser.add_argument('--ci', type=float, default=0.90)
    parser.add_argument('--flood_threshold_cm', type=float, default=5.0)
    parser.add_argument('--cp_eps_cm', type=float, default=0.5, help='Floor on std in the cp_norm score')
    parser.add_argument('--gen_batch', type=int, default=40, help='Scenes per batch when generating train latents')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-debug', '-d', action='store_true')
    args = parser.parse_args()
    args.phase = 'test'
    return args


# ------------------------------------------------------------------ SSF adapter

class SSFDecoder:
    """Per-channel scale and shift on the output of every Conv2d / Linear in a module,
    applied through forward hooks, so autoencoder.py stays untouched. With gamma=1, beta=0
    it is an exact identity, and while disabled the hooks do nothing."""

    def __init__(self, module):
        self.layers = [m for m in module.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        self.device = next(module.parameters()).device
        self.gammas = [torch.ones(self._channels(m), device=self.device) for m in self.layers]
        self.betas = [torch.zeros(self._channels(m), device=self.device) for m in self.layers]
        self.enabled = False
        self.handles = [m.register_forward_hook(self._hook(i)) for i, m in enumerate(self.layers)]

    @staticmethod
    def _channels(m):
        return m.out_channels if isinstance(m, nn.Conv2d) else m.out_features

    def _hook(self, i):
        def hook(module, inputs, out):
            if not self.enabled:
                return None
            g, b = self.gammas[i].to(out.dtype), self.betas[i].to(out.dtype)
            if isinstance(module, nn.Conv2d):  # (B, C, H, W); Linear outputs are channels-last
                g, b = g.view(1, -1, 1, 1), b.view(1, -1, 1, 1)
            return out * g + b
        return hook

    def n_params(self):
        return 2 * sum(g.numel() for g in self.gammas)

    def parameters(self):
        return self.gammas + self.betas

    def reset_identity(self):
        self.load({'gamma': [torch.ones_like(g) for g in self.gammas],
                   'beta': [torch.zeros_like(b) for b in self.betas]})

    def init_member(self, gen, init_std):
        self.gammas = [(1 + init_std * torch.randn(g.shape, generator=gen)).to(self.device).requires_grad_()
                       for g in self.gammas]
        self.betas = [(init_std * torch.randn(b.shape, generator=gen)).to(self.device).requires_grad_()
                      for b in self.betas]

    def state(self):
        return {'gamma': [g.detach().cpu().clone() for g in self.gammas],
                'beta': [b.detach().cpu().clone() for b in self.betas]}

    def load(self, state):
        self.gammas = [g.to(self.device) for g in state['gamma']]
        self.betas = [b.to(self.device) for b in state['beta']]


# ------------------------------------------------------------------ scoring

def interval_metrics(lo, hi, y, ci):
    """Flattened tensors of lower bound, upper bound and truth (cm). Interval score is the
    proper score for a central (ci) interval: width plus 2/alpha times any miss distance."""
    alpha = 1 - ci
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    score = width + (2 / alpha) * ((lo - y).clamp(min=0) + (y - hi).clamp(min=0))
    return {'coverage': inside.float().mean().item(),
            'mean_pi_width_cm': width.mean().item(),
            'interval_score_cm': score.mean().item()}


def ensemble_crps(ens, y, chunk=1_000_000):
    """Mean CRPS of a (K, P) ensemble against (P,) truth: E|X - y| - 0.5 E|X - X'|,
    with the pairwise term in O(K log K) from the sorted members."""
    K = ens.shape[0]
    w = (2 * torch.arange(1, K + 1, dtype=ens.dtype) - K - 1)
    total = 0.0
    for s in range(0, ens.shape[1], chunk):
        x, t = ens[:, s:s + chunk], y[s:s + chunk]
        term1 = (x - t).abs().mean(0)
        pair = (w[:, None] * x.sort(0).values).sum(0) * 2 / K ** 2  # E|X - X'|
        total += (term1 - 0.5 * pair).sum().item()
    return total / ens.shape[1]


def ensemble_metrics(ens, y, ci):
    """(K, P) masked ensemble vs (P,) truth: the usual quantile PI plus CRPS and spread/skill."""
    alpha = 1 - ci
    lo = torch.quantile(ens, alpha / 2, dim=0)
    hi = torch.quantile(ens, 1 - alpha / 2, dim=0)
    mean = ens.mean(0)
    spread = torch.sqrt((ens.std(0, unbiased=ens.shape[0] > 1) ** 2).mean()).item()
    rmse = torch.sqrt(((mean - y) ** 2).mean()).item()
    out = interval_metrics(lo, hi, y, ci)
    out.update({'ensemble_mean_rmse_cm': rmse, 'spread_cm': spread,
                'spread_skill_ratio': spread / rmse if rmse > 0 else float('nan'),
                'crps_cm': ensemble_crps(ens, y), 'n_samples': int(ens.shape[0]), 'n_pixels': int(ens.shape[1])})
    return out


def _conformal_q(scores, ci):
    n = scores.numel()
    level = min(1.0, math.ceil((n + 1) * ci) / n)
    return float(np.quantile(scores.numpy(), level, method='higher'))


def conformal(cal_ens, cal_y, ev_ens, ev_y, ci, kind, eps):
    """Split conformal from a calibration ensemble (Kc, Pc) onto an eval ensemble (K, P)."""
    alpha = 1 - ci
    if kind == 'cqr':
        def bounds(e):
            return torch.quantile(e, alpha / 2, dim=0), torch.quantile(e, 1 - alpha / 2, dim=0)
        clo, chi = bounds(cal_ens)
        q = _conformal_q(torch.maximum(clo - cal_y, cal_y - chi), ci)
        lo, hi = bounds(ev_ens)
        lo, hi = lo - q, hi + q
    else:
        cm, em = cal_ens.mean(0), ev_ens.mean(0)
        if kind == 'abs':
            q = _conformal_q((cal_y - cm).abs(), ci)
            half = torch.full_like(em, q)
        elif kind == 'norm':
            cs = cal_ens.std(0, unbiased=cal_ens.shape[0] > 1) + eps
            q = _conformal_q((cal_y - cm).abs() / cs, ci)
            half = q * (ev_ens.std(0, unbiased=ev_ens.shape[0] > 1) + eps)
        else:
            raise ValueError(kind)
        lo, hi = em - half, em + half
    out = interval_metrics(lo.clamp(min=0), hi, ev_y, ci)
    out.update({'q': q, 'ensemble_mean_rmse_cm': torch.sqrt(((ev_ens.mean(0) - ev_y) ** 2).mean()).item()})
    return out


# ------------------------------------------------------------------ main

def main():
    import core.logger as Logger
    import core.metrics as Metrics
    import data as Data
    import model as Model
    from collapse_ablation import Runner, _err_stats
    from coverage_diagnostic import coverage_and_stats

    args = parse_args()
    opt = Logger.dict_to_nonedict(Logger.parse(args))
    logger = Logger.setup_logger('base', opt['path']['log'], 'decoder_uq', screen=True)
    logger.info(Logger.dict2str(opt))
    logger.info(f'decoder_uq args: {vars(args)}')
    torch.backends.cudnn.benchmark = True
    if opt['model']['which_model_G'] != 'ddpm':
        raise SystemExit('decoder_uq needs a latent (ddpm) config: there is no decoder to adapt in pixel space')

    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['test'], schedule_phase='test')
    runner = Runner(diffusion, opt, diffusion.device)
    dev = diffusion.device
    bs_test = runner.resolve_bs(opt['model']['beta_schedule']['test'].get('backward_start') or 'full')

    meta = opt['datasets']['meta']
    norm_range = (meta['norm_min'], meta['norm_max'])
    to_depth = lambda x: Metrics.unnormalize(x.float().cpu().clamp(*norm_range), meta['max_depth'], min_max=norm_range)
    thr_norm = norm_range[0] + args.flood_threshold_cm / meta['max_depth'] * (norm_range[1] - norm_range[0])

    base_opt = dict(opt['datasets']['test'], data_len=-1, use_shuffle=False)
    test_set = Data.create_dataset(base_opt, 'test', meta, latent=False, dem=opt['dem'])
    train_set = Data.create_dataset(base_opt, 'train', meta, latent=False, dem=opt['dem'])
    rng = np.random.default_rng(args.seed)
    eval_idx = list(range(min(args.n_eval, len(test_set))))
    rest = np.arange(len(eval_idx), len(test_set))
    calib_idx = sorted(rng.choice(rest, size=min(args.n_calib, len(rest)), replace=False).tolist())
    train_idx = sorted(rng.choice(len(train_set), size=min(args.n_train, len(train_set)), replace=False).tolist())

    def loader(ds, idx, bs):
        return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=bs,
                                           shuffle=False, num_workers=2, pin_memory=True)

    def one_batch(ds, idx):
        return next(iter(loader(ds, idx, len(idx))))

    def draw_latents(batch, n_draws, seed):
        # With seed=args.seed this is collapse_ablation's seeding and draw order, so `base` on
        # the eval set reproduces its numbers.
        lr, dem = batch['SR'].to(dev), batch['DEM'].to(dev) if 'DEM' in batch else None
        torch.manual_seed(seed)
        return torch.stack([runner.sample_latent(lr, dem, bs_test, 'sample').cpu() for _ in range(n_draws)], 0)

    results = {'args': vars(args), 'config': args.config, 'backward_start': bs_test,
               'eval_files': None, 'calib_files': None}
    out_dir = opt['path']['results']
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'decoder_uq.json')

    def dump():
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    # ---------------------------------------------------------- latent pools
    t0 = time.time()
    ev_batch, ca_batch = one_batch(test_set, eval_idx), one_batch(test_set, calib_idx)
    results['eval_files'], results['calib_files'] = list(ev_batch['filename']), list(ca_batch['filename'])
    ev_lat = draw_latents(ev_batch, args.n_samples, args.seed)
    ca_lat = draw_latents(ca_batch, args.n_samples, args.seed + 2)  # independent noise from eval
    ev_hr, ca_hr = to_depth(ev_batch['HR']), to_depth(ca_batch['HR'])
    ev_mask, ca_mask = ev_hr > args.flood_threshold_cm, ca_hr > args.flood_threshold_cm
    ev_mu = runner.encode(ev_batch['HR'].to(dev))[1].cpu()
    del ev_batch, ca_batch
    logger.info(f'eval {len(eval_idx)} scenes (flooded {ev_mask.float().mean():.1%}), calib {len(calib_idx)} '
                f'scenes (flooded {ca_mask.float().mean():.1%}), {args.n_samples} latents each, '
                f'backward_start={bs_test} ({time.time() - t0:.0f}s)')

    t0 = time.time()
    tr_lat, tr_hr = [], []
    torch.manual_seed(args.seed + 1)
    for b in loader(train_set, train_idx, args.gen_batch):
        hr = b['HR'].to(dev)
        if args.train_target == 'ldm':
            dem = b['DEM'].to(dev) if 'DEM' in b else None
            tr_lat.append(runner.sample_latent(b['SR'].to(dev), dem, bs_test, 'sample').cpu())
        else:
            tr_lat.append(runner.encode(hr)[1].cpu())
        tr_hr.append(b['HR'].float())
    tr_lat, tr_hr = torch.cat(tr_lat), torch.cat(tr_hr)
    logger.info(f'train pool: {tr_lat.shape[0]} scenes, target={args.train_target}, '
                f'flooded {(tr_hr > thr_norm).float().mean():.1%} ({time.time() - t0:.0f}s)')

    # ---------------------------------------------------------- helpers
    adapter = SSFDecoder(runner.net.vae.decoder)
    results['ssf_params'] = adapter.n_params()
    logger.info(f'SSF on VAE decoder: {len(adapter.layers)} layers, {adapter.n_params():,} params per member')

    def decode_stack(lat, member_of=None, states=None):
        """Decode (K, N, 4, h, w) latents. member_of(k) picks the SSF state for draw k (None = base)."""
        out = []
        for k in range(lat.shape[0]):
            if member_of is None:
                adapter.enabled = False
            else:
                adapter.load(states[member_of(k)])
                adapter.enabled = True
            out.append(to_depth(runner.decode(lat[k].to(dev))))
        adapter.enabled = False
        return torch.stack(out, 0)

    def flooded_rmse_of_mu(state=None):
        if state is None:
            adapter.enabled = False
        else:
            adapter.load(state)
            adapter.enabled = True
        rec = to_depth(runner.decode(ev_mu.to(dev)))
        adapter.enabled = False
        return _err_stats(rec, ev_hr, ev_mask)['rmse_cm']

    # An identity adapter must reproduce the frozen decoder. A tolerance rather than exact
    # equality, because repeated cuDNN forward passes aren't guaranteed bit-identical.
    adapter.reset_identity()
    z = ev_lat[0, :2].to(dev)
    ref = runner.decode(z)
    adapter.enabled = True
    diff = (ref - runner.decode(z)).abs().max().item()
    adapter.enabled = False
    logger.info(f'identity-SSF check: max |diff| {diff:.2e} (normalised units)')
    if diff > 1e-3:
        raise SystemExit('SSF identity check failed; hooks change the decoder output')

    results['vae_floor_flooded_rmse_cm'] = {'base': flooded_rmse_of_mu()}
    dump()

    # ---------------------------------------------------------- train SSF members
    amp_dtype = torch.bfloat16 if dev.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    states, train_log = [], []
    for m in range(args.n_members):
        gen = torch.Generator().manual_seed(args.seed * 1000 + m + 1)
        adapter.init_member(gen, args.init_std)
        adapter.enabled = True
        n = tr_lat.shape[0]
        pool = torch.arange(n) if args.no_bootstrap else torch.randint(n, (n,), generator=gen)
        optim = torch.optim.Adam(adapter.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.steps)
        t0, losses = time.time(), []
        for step in range(args.steps):
            pick = pool[torch.randint(len(pool), (args.batch_size,), generator=gen)]
            z, y = tr_lat[pick].to(dev), tr_hr[pick].to(dev)
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=dev.type == 'cuda'):
                pred = runner.net.vae.decode(z)
            w = 1 + (args.flood_weight - 1) * (y > thr_norm).float()
            loss = (w * (pred.float() - y) ** 2).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()
            losses.append(loss.item())
            if step == 0 or (step + 1) % 250 == 0:
                logger.info(f'  member {m}: step {step + 1}/{args.steps} loss {np.mean(losses[-250:]):.3e} '
                            f'({time.time() - t0:.0f}s)')
        adapter.enabled = False
        states.append(adapter.state())
        torch.save(states[-1], os.path.join(out_dir, f'ssf_member{m}.pt'))
        floor = flooded_rmse_of_mu(states[-1])
        train_log.append({'member': m, 'final_loss': float(np.mean(losses[-250:])),
                          'seconds': time.time() - t0, 'vae_floor_flooded_rmse_cm': floor})
        logger.info(f'member {m} done: VAE floor (flooded RMSE, eval scenes) {floor:.3f} cm '
                    f"vs base {results['vae_floor_flooded_rmse_cm']['base']:.3f} cm")
        results['ssf_training'] = train_log
        dump()

    # ---------------------------------------------------------- evaluate
    ci, M = args.ci, args.n_members
    members = lambda k: k % M
    flooded = lambda stack, mask: stack[:, mask]

    t0 = time.time()
    ev_base = decode_stack(ev_lat)
    ev_ssf = decode_stack(ev_lat, members, states)
    ev_ssf1 = decode_stack(ev_lat, lambda k: 0, states)
    ca_base = decode_stack(ca_lat)
    ca_ssf = decode_stack(ca_lat, members, states)
    logger.info(f'decoded all stacks ({time.time() - t0:.0f}s)')

    y_ev, y_ca = ev_hr[ev_mask], ca_hr[ca_mask]
    E = {'base': flooded(ev_base, ev_mask), 'ssf_single': flooded(ev_ssf1, ev_mask),
         'ssf_ens': flooded(ev_ssf, ev_mask)}
    C = {'base': flooded(ca_base, ca_mask), 'ssf_ens': flooded(ca_ssf, ca_mask)}

    rows = {}
    for name, ens in E.items():
        rows[name] = ensemble_metrics(ens, y_ev, ci)
    # Same function as the earlier runs, as a cross-check that `base` reproduces them
    results['base_coverage_and_stats_flooded'] = coverage_and_stats(ev_base, ev_hr, ci, mask=ev_mask)
    for kind in ('abs', 'norm', 'cqr'):
        rows[f'cp_{kind}'] = conformal(C['base'], y_ca, E['base'], y_ev, ci, kind, args.cp_eps_cm)
    rows['ssf_ens+cp_norm'] = conformal(C['ssf_ens'], y_ca, E['ssf_ens'], y_ev, ci, 'norm', args.cp_eps_cm)
    results['eval_flooded'] = rows
    dump()

    logger.info('=' * 78)
    logger.info(f'FLOODED pixels, {len(eval_idx)} eval scenes, nominal {ci:.0%} interval '
                f'(interval score and CRPS: lower is better)')
    logger.info(f"{'method':>16} {'cov':>6} {'width':>7} {'IS':>7} {'CRPS':>6} {'RMSE':>7} {'ss':>6}")
    for name, r in rows.items():
        crps = f"{r['crps_cm']:6.3f}" if 'crps_cm' in r else f"{'-':>6}"
        ss = f"{r['spread_skill_ratio']:6.3f}" if 'spread_skill_ratio' in r else f"{'-':>6}"
        logger.info(f"{name:>16} {r['coverage']:6.3f} {r['mean_pi_width_cm']:7.3f} "
                    f"{r['interval_score_cm']:7.3f} {crps} {r['ensemble_mean_rmse_cm']:7.3f} {ss}")
    logger.info('(width, IS, CRPS, RMSE in cm)')
    floors = [t['vae_floor_flooded_rmse_cm'] for t in train_log]
    logger.info(f"VAE floor flooded RMSE: base {results['vae_floor_flooded_rmse_cm']['base']:.3f} cm, "
                f"SSF members {min(floors):.3f}-{max(floors):.3f} cm")
    dump()
    logger.info(f'Saved {out_path}')


if __name__ == '__main__':
    main()
