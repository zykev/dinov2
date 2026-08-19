#!/usr/bin/env python
"""Wrap the official DINOv2 plain state_dict into {"model": ...} for fvcore Checkpointer.

Also bicubic-interpolates pos_embed from the source resolution (e.g. 37x37 @518px)
to the target grid (e.g. 16x16 @224px), mirroring interpolate_pos_encoding.
"""
import argparse
import math

import torch
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="official plain state_dict .pth")
    ap.add_argument("--dst", required=True, help="output wrapped checkpoint path")
    ap.add_argument("--global-crops-size", type=int, default=224)
    ap.add_argument("--patch-size", type=int, default=14)
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")
    assert isinstance(ckpt, dict) and "model" not in ckpt, \
        "official checkpoint should be a plain state_dict"

    h0 = w0 = args.global_crops_size // args.patch_size
    if "pos_embed" in ckpt:
        pe = ckpt["pos_embed"]
        dim = pe.shape[-1]
        M = int(math.sqrt(pe.shape[1] - 1))
        assert M * M == pe.shape[1] - 1, f"pos_embed grid not square: {pe.shape}"
        if M != h0:
            patch_pe = pe[:, 1:].reshape(1, M, M, dim).permute(0, 3, 1, 2)
            patch_pe = F.interpolate(patch_pe, size=(h0, w0), mode="bicubic", antialias=False)
            ckpt["pos_embed"] = torch.cat(
                (pe[:, 0:1], patch_pe.permute(0, 2, 3, 1).reshape(1, -1, dim)), dim=1
            )
            print(f"[wrap_checkpoint] interpolated pos_embed {M}x{M} -> {h0}x{w0}")

    torch.save({"model": ckpt}, args.dst)
    print(f"[wrap_checkpoint] saved wrapped checkpoint ({len(ckpt)} keys) -> {args.dst}")


if __name__ == "__main__":
    main()
