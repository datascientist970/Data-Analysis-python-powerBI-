from __future__ import annotations
import argparse
import glob
import os

from .pipeline import run_pipeline


def parse_args():
    p = argparse.ArgumentParser(description='Hybrid Affordance Detection System')
    p.add_argument('--images', type=str, required=True, help='Path to image file or glob pattern')
    p.add_argument('--model', type=str, required=True, help='Path to trained supervised model .pth')
    p.add_argument('--out_dir', type=str, default='/hybrid_results', help='Output directory')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--spm', type=str, default='pca', choices=['pca', 'ae', 'vae'])
    p.add_argument('--spm_components', type=int, default=32)
    p.add_argument('--mcm', type=str, default='isomap', choices=['umap', 'isomap', 'spectral'])
    p.add_argument('--mcm_components', type=int, default=2)
    p.add_argument('--cluster', type=str, default='dbscan', choices=['dbscan', 'hdbscan'])
    p.add_argument('--metric', type=str, default='euclidean', choices=['euclidean', 'cosine'])
    p.add_argument('--fusion', type=str, default='hard', choices=['hard', 'soft'])
    p.add_argument('--conf_thresh', type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()

    image_paths = []
    if any(ch in args.images for ch in ['*', '?', '[']):
        image_paths = sorted(glob.glob(args.images))
    elif os.path.isdir(args.images):
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            image_paths.extend(glob.glob(os.path.join(args.images, ext)))
        image_paths = sorted(image_paths)
    else:
        image_paths = [args.images]

    if len(image_paths) == 0:
        raise FileNotFoundError(f'No images found for pattern/path: {args.images}')

    os.makedirs(args.out_dir, exist_ok=True)

    for img in image_paths:
        print(f"Processing: {img}")
        run_pipeline(
            image_path=img,
            model_path=args.model,
            out_dir=args.out_dir,
            image_size=args.image_size,
            spm=args.spm,
            spm_components=args.spm_components,
            mcm=args.mcm,
            mcm_components=args.mcm_components,
            cluster=args.cluster,
            metric=args.metric,
            fusion_mode=args.fusion,
            confidence_threshold=args.conf_thresh,
        )
        print(f"Done: {img}")


if __name__ == '__main__':
    main()
