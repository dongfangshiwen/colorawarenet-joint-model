"""Training implementation used by train.py and the package CLI."""
from copy import deepcopy
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.datasets import make_dataset
from ..data.splits import training_split
from ..losses import VGGPerceptualLoss, dehaze_losses, segmentation_losses
from ..metrics import (AverageMeter, batch_psnr, batch_ssim, saturation_map,
                       chromaticity_ratio_error, confusion_matrix, segmentation_metrics)
from ..models.joint import JointDehazeSegModel, unpack_dehaze_output
from ..models.DCP import ClassicalDCP
from ..models.registry import build_model, model_config
from ..utils import ensure_dir, select_device, set_seed, write_csv, write_json
from .checkpoint import save_checkpoint


def configure_stage(model, stage):
    """Freeze parameters AND running statistics of the inactive branch."""
    joint = isinstance(model, JointDehazeSegModel)
    if stage not in ("dehaze", "seg", "joint") or (not joint and stage != "dehaze"):
        raise ValueError(f"Stage {stage!r} is incompatible with this model")
    dehazer = model.dehazer if joint else model
    if isinstance(dehazer, ClassicalDCP) and stage != "seg":
        raise ValueError("Classical DCP stays fixed; only its downstream segmentation stage can be trained")
    model.train()
    if joint:
        for module, active in ((model.dehazer, stage != "seg"), (model.segmenter, stage != "dehaze")):
            module.train(active)
            for parameter in module.parameters():
                parameter.requires_grad_(active)
                parameter.grad = None
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)


def train_epoch(model, loader, optimizer, device, args, stage, perceptual=None, scaler=None):
    configure_stage(model, stage)
    meters = {}
    for batch in tqdm(loader, desc=stage, leave=False):
        hazy, clear = batch["hazy"].to(device), batch["clear"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            if stage == "dehaze":
                dehazer = model.dehazer if isinstance(model, JointDehazeSegModel) else model
                restored, _ = unpack_dehaze_output(dehazer(hazy))
                losses = dehaze_losses(restored, clear, args, perceptual)
                total = args.lam_dehaze * losses["dehaze"]
            elif stage == "seg":
                with torch.no_grad():
                    restored, _ = unpack_dehaze_output(model.dehazer(hazy))
                logits = model.segment(restored)
                losses = segmentation_losses(logits, batch["mask"].to(device), args)
                total = args.lam_seg * losses["seg"]
            else:
                restored, logits, _ = model(hazy)
                losses = dehaze_losses(restored, clear, args, perceptual)
                losses.update(segmentation_losses(logits, batch["mask"].to(device), args))
                total = args.lam_dehaze * losses["dehaze"] + args.lam_seg * losses["seg"]
        if not torch.isfinite(total):
            raise RuntimeError(f"Non-finite {stage} loss for samples {batch['id']}")
        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
        else:
            total.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        for key, value in dict(total=total, **losses).items():
            meters.setdefault(key, AverageMeter()).update(value.detach().item(), len(hazy))
    return {key: meter.avg for key, meter in meters.items()}


@torch.no_grad()
def image_metrics(pred, target):
    sat_pred, sat_ref = saturation_map(pred), saturation_map(target)
    return dict(psnr=batch_psnr(pred, target), ssim=batch_ssim(pred, target),
                l1=float((pred - target).abs().mean()), mse=float((pred - target).square().mean()),
                delta_sat=float((sat_pred - sat_ref).abs().mean()),
                crerr=chromaticity_ratio_error(pred, target),
                sat_pred=float(sat_pred.mean()), sat_ref=float(sat_ref.mean()))


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    meters, cm = {}, None
    for batch in loader:
        hazy, clear = batch["hazy"].to(device), batch["clear"].to(device)
        if isinstance(model, JointDehazeSegModel):
            pred, logits, _ = model(hazy)
            if "mask" in batch:
                current = confusion_matrix(logits.argmax(1), batch["mask"].to(device), logits.shape[1])
                cm = current if cm is None else cm + current
        else:
            pred, _ = unpack_dehaze_output(model(hazy))
        # Per-image averaging avoids PSNR changing with the validation batch size.
        for p, t in zip(pred, clear):
            for key, value in image_metrics(p[None], t[None]).items():
                meters.setdefault(key, AverageMeter()).update(value)
    result = {key: meter.avg for key, meter in meters.items()}
    if cm is not None:
        result.update(segmentation_metrics(cm))
    return result


def stage_schedule(args, strategy="full_staged"):
    if args.model == "dcp" and args.dcp_mode == "classical":
        if args.dataset != "paired-road":
            raise ValueError("DCP classical training requires --dataset paired-road with hazy/, clear/ and masks/: "
                             "only the downstream segmenter is trainable. For benchmark restoration training, "
                             "use --dcp-mode learned; for classical evaluation, use evaluate --model dcp.")
        return [("seg", args.pretrain_seg_epochs + args.finetune_epochs)]
    if args.dataset != "paired-road":
        return [("dehaze", args.pretrain_dehaze_epochs)]
    if strategy == "direct_joint":
        return [("joint", args.direct_joint_epochs)]
    if strategy == "seg_only":
        return [("seg", args.pretrain_seg_epochs + args.finetune_epochs)]
    stages = [("dehaze", args.pretrain_dehaze_epochs)]
    if strategy != "dehaze_only":
        stages.append(("seg", args.pretrain_seg_epochs))
    if strategy == "full_staged":
        stages.append(("joint", args.finetune_epochs))
    return stages


def run_training(args, experiment=None, experiment_name=None):
    args = deepcopy(args)
    experiment = experiment or {}
    set_seed(args.seed)
    device = select_device(args.device)
    config = model_config(args.model, joint=args.dataset == "paired-road", dcp_mode=args.dcp_mode)
    if args.model == "coloraware":
        config["dehazer"]["base_ch"] = args.color_base_ch
    config["segmenter"].update(num_classes=args.num_classes, base_ch=args.seg_base_ch,
                               width_mult=args.seg_width_mult, attention=not args.no_attention,
                               use_se=args.use_se)
    config["imagenet_norm"] = not args.no_imagenet_norm
    config["dehazer"].update(experiment.get("dehazer", {}))
    config["segmenter"].update(experiment.get("segmenter", {}))
    config["imagenet_norm"] = experiment.get("imagenet_norm", config["imagenet_norm"])
    config["dehazer_type"] = experiment.get("dehazer_type", config["dehazer_type"])
    for key, value in experiment.get("loss", {}).items():
        setattr(args, key, value)
    stages = [(s, n) for s, n in stage_schedule(args, experiment.get("strategy", "full_staged")) if n > 0]
    if not stages:
        if args.model == "dcp" and args.dcp_mode == "classical":
            raise ValueError("DCP requires --pretrain-seg-epochs + --finetune-epochs > 0")
        raise ValueError("At least one training stage must have a positive epoch count")
    train_config = dict(vars(args),
                        training_protocol=("fixed-dcp-segmentation" if args.model == "dcp" and args.dcp_mode == "classical" else
                                           "learned-dcp-joint" if args.model == "dcp" and config["joint"] else
                                           "learned-dcp-restoration" if args.model == "dcp" else
                                           "restoration-only" if not config["joint"] else
                                           experiment.get("strategy", "full_staged")),
                        stages=[dict(stage=s, epochs=n) for s, n in stages])
    save_dir = Path(args.output) / args.dataset / (experiment_name or args.model)
    if (save_dir / "metrics.csv").exists() or (save_dir / "last.pth").exists():
        raise ValueError(f"An experiment already exists in {save_dir}; choose another --output")
    ds_kwargs = dict(name=args.dataset, root=args.data_root, resize=tuple(args.resize),
                     num_classes=args.num_classes, train_repeats=args.train_repeats)
    full = make_dataset(**ds_kwargs)
    val_kwargs = dict(ds_kwargs, root=args.val_root or args.data_root)
    val_full = make_dataset(**val_kwargs) if args.val_root else None
    data_split = training_split(args.dataset, args.data_root, full.ids, args.val_ratio, args.seed,
                                args.val_root, val_full.ids if val_full is not None else None, args.split_unit)
    if data_split["shared_clear_references"]:
        print(f"Warning: validation shares clear references with training: {data_split['shared_clear_references']}. "
              "Use --split-unit scene for independent reference groups; sample splits are historical diagnostics.")
    train_ids, val_ids = data_split["train"], data_split["val"]
    train_ds = make_dataset(**ds_kwargs, ids=train_ids, augment=experiment.get("augment", not args.no_augment))
    val_ds = make_dataset(**val_kwargs, ids=val_ids, augment=False)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, shuffle=True, **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_options)
    model = build_model(config).to(device)
    needs_dehaze = any(s != "seg" for s, _ in stages)
    perceptual = VGGPerceptualLoss(device) if needs_dehaze and args.lam_perc > 0 else None
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    ensure_dir(save_dir)
    write_json(save_dir / "split.json", data_split)
    write_json(save_dir / "config.json", dict(model_config=config, train_config=train_config, experiment=experiment))
    print(f"Device: {device}; train {len(train_ids)}, validation {len(val_ids)}; split: {data_split['grouping']}")
    if args.model == "dcp" and args.dcp_mode == "classical":
        print(f"DCP remains fixed; training LiteAttentionUNet for {stages[0][1]} epochs with CE + Dice. "
              "--pretrain-dehaze-epochs is unused; no dehazing or joint optimization stage is run.")
    elif args.model == "dcp":
        print("DCP uses trainable refinement after analytic recovery; "
              f"schedule: {', '.join(f'{stage} {epochs}' for stage, epochs in stages)}. "
              "This is the learned DCP extension.")
    rows, final_stats = [], {}
    for stage, epochs in stages:
        configure_stage(model, stage)
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError(f"No trainable parameters in stage {stage}")
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=5)
        best = float("-inf")
        for epoch in range(1, epochs + 1):
            started = time.monotonic()
            train_stats = train_epoch(model, train_loader, optimizer, device, args, stage, perceptual, scaler)
            final_stats = validate(model, val_loader, device)
            metric = "psnr" if stage == "dehaze" else ("miou" if stage == "seg" else args.score_metric)
            score = (final_stats["psnr"] / 30 + final_stats["ssim"] + final_stats["miou"] + final_stats["f1"]
                     if metric == "balanced" else final_stats[metric])
            scheduler.step(score)
            improved = score > best
            best = max(best, score)
            row = dict(stage=stage, epoch=epoch, global_epoch=len(rows) + 1, lr=optimizer.param_groups[0]["lr"],
                       score_metric=metric, score=score,
                       **{"train_" + k: v for k, v in train_stats.items()},
                       **{"val_" + k: v for k, v in final_stats.items() if isinstance(v, (int, float))})
            for key in ("class_iou", "class_dice"):
                row.update({f"val_{key}_{i}": v for i, v in enumerate(final_stats.get(key, []))})
            rows.append(row)
            write_csv(save_dir / "metrics.csv", rows)
            save_args = (model, config, train_config, stage, epoch, final_stats, optimizer, scheduler, scaler)
            save_checkpoint(save_dir / stage / "last.pth", *save_args, data_split=data_split)
            if improved:
                save_checkpoint(save_dir / stage / "best.pth", *save_args, data_split=data_split)
            if stage == stages[-1][0]:
                save_checkpoint(save_dir / "last.pth", *save_args, data_split=data_split)
                if improved:
                    save_checkpoint(save_dir / "best.pth", *save_args, data_split=data_split)
            print(f"[{stage} {epoch}/{epochs}] loss={train_stats['total']:.5f} {metric}={score:.5f} ({time.monotonic()-started:.1f}s)")
    write_json(save_dir / "final_metrics.json", final_stats)
    if not args.no_plots:
        from ..visualization.curves import save_metric_plots_from_csv
        save_metric_plots_from_csv(save_dir / "metrics.csv", save_dir)
    print(f"Saved: {save_dir}")
    return final_stats
