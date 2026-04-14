import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _iter_images(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=lambda p: _natural_key(p.name))


def _natural_key(text: str):
    # Sort 1,2,10 in numeric order instead of lexicographic order.
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', text)]


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'hardlink':
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)
    else:
        shutil.copy2(src, dst)


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _collect_scene_pairs(sharp_root: Path, blur_root: Path) -> List[Tuple[str, List[Tuple[Path, Path]]]]:
    sharp_scenes = sorted([p for p in sharp_root.iterdir() if p.is_dir()], key=lambda p: _natural_key(p.name))
    blur_scenes = sorted([p for p in blur_root.iterdir() if p.is_dir()], key=lambda p: _natural_key(p.name))
    sharp_names = {p.name for p in sharp_scenes}
    blur_names = {p.name for p in blur_scenes}
    common = sorted(sharp_names & blur_names, key=_natural_key)

    if not common:
        raise RuntimeError(f'No common scene folders found between {sharp_root} and {blur_root}.')

    missing_in_blur = sorted(sharp_names - blur_names)
    missing_in_sharp = sorted(blur_names - sharp_names)
    if missing_in_blur or missing_in_sharp:
        print('[warn] Scene mismatch detected.')
        if missing_in_blur:
            print(f'  only in sharp: {missing_in_blur}')
        if missing_in_sharp:
            print(f'  only in blur:  {missing_in_sharp}')

    pairs: List[Tuple[str, List[Tuple[Path, Path]]]] = []
    for scene in common:
        s_dir = sharp_root / scene
        b_dir = blur_root / scene
        s_imgs = _iter_images(s_dir)
        b_imgs = _iter_images(b_dir)

        s_map: Dict[str, Path] = {p.name: p for p in s_imgs}
        b_map: Dict[str, Path] = {p.name: p for p in b_imgs}
        names = sorted(set(s_map.keys()) & set(b_map.keys()), key=_natural_key)
        if not names:
            raise RuntimeError(f'No paired files in scene {scene}.')

        only_s = sorted(set(s_map.keys()) - set(b_map.keys()))
        only_b = sorted(set(b_map.keys()) - set(s_map.keys()))
        if only_s or only_b:
            raise RuntimeError(
                f'Unpaired files in scene {scene}. '
                f'only_sharp={len(only_s)}, only_blur={len(only_b)}')

        scene_pairs: List[Tuple[Path, Path]] = []
        for name in names:
            scene_pairs.append((s_map[name], b_map[name]))
        pairs.append((scene, scene_pairs))

    return pairs


def _flatten_split(data_root: Path, split: str, mode: str, dry_run: bool) -> None:
    sharp_root = data_root / f'{split}_sharp'
    blur_root = data_root / f'{split}_blur'
    mask_root = data_root / f'{split}_mask'

    out_sharp = data_root / f'{split}_sharp_flat'
    out_blur = data_root / f'{split}_blur_flat'
    out_mask = data_root / f'{split}_mask_flat'

    if not sharp_root.exists() or not blur_root.exists():
        print(f'[skip] {split}: missing {sharp_root} or {blur_root}')
        return

    pairs = _collect_scene_pairs(sharp_root, blur_root)

    if not dry_run:
        # Force overwrite behavior on every run to avoid stale files.
        _reset_output_dir(out_sharp)
        _reset_output_dir(out_blur)
        _reset_output_dir(out_mask)

    copied = 0
    used_names = set()
    split_l = split.lower()

    def write_pair(sharp_path: Path, blur_path: Path, out_name: str) -> None:
        nonlocal copied
        if out_name in used_names:
            raise RuntimeError(f'Duplicate output filename detected in {split}: {out_name}')
        used_names.add(out_name)

        if not dry_run:
            _copy_or_link(sharp_path, out_sharp / out_name, mode)
            _copy_or_link(blur_path, out_blur / out_name, mode)
        copied += 1

    if split_l in {'val', 'test'}:
        if len(pairs) != 1:
            raise RuntimeError(
                f'{split} expects exactly one scene folder for no-rename flattening, got {len(pairs)} scenes.')
        _, scene_pairs = pairs[0]
        for sharp_path, blur_path in scene_pairs:
            # Keep original filenames for val/test.
            write_pair(sharp_path, blur_path, sharp_path.name)
    elif split_l == 'train':
        # Renumber continuously across scenes: 001(1..N), 002(N+1..), ...
        global_idx = 1
        for _, scene_pairs in pairs:
            for sharp_path, blur_path in scene_pairs:
                write_pair(sharp_path, blur_path, f'{global_idx}{sharp_path.suffix}')
                global_idx += 1
    else:
        # Fallback for custom splits: keep scene prefix to avoid collisions.
        for scene, scene_pairs in pairs:
            for sharp_path, blur_path in scene_pairs:
                write_pair(sharp_path, blur_path, f'{scene}_{sharp_path.name}')

    # Also flatten scene-level mask files for traceability.
    if mask_root.exists() and not dry_run:
        scene_dirs = sorted([p for p in mask_root.iterdir() if p.is_dir()], key=lambda p: _natural_key(p.name))
        for scene_dir in scene_dirs:
            csv_file = scene_dir / 'blind_pixel_coords.csv'
            mask_img = scene_dir / 'blind_pixel_mask.png'
            if split_l in {'val', 'test'} and len(scene_dirs) == 1:
                csv_name = 'blind_pixel_coords.csv'
                mask_name = 'blind_pixel_mask.png'
            else:
                csv_name = f'{scene_dir.name}_blind_pixel_coords.csv'
                mask_name = f'{scene_dir.name}_blind_pixel_mask.png'
            if csv_file.exists():
                _copy_or_link(csv_file, out_mask / csv_name, 'copy')
            if mask_img.exists():
                _copy_or_link(mask_img, out_mask / mask_name, 'copy')

    print(f'[ok] {split}: paired images = {copied}')
    if not dry_run:
        print(f'     sharp -> {out_sharp}')
        print(f'     blur  -> {out_blur}')
        if mask_root.exists():
            print(f'     mask  -> {out_mask}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Flatten scene-based paired datasets (001/002/...) into flat *_sharp_flat/*_blur_flat folders.')
    parser.add_argument('--data_root', required=True, help='Dataset root that contains train_blur/train_sharp etc.')
    parser.add_argument('--splits', default='train,val,test', help='Comma-separated splits to process.')
    parser.add_argument('--mode', choices=['copy', 'hardlink'], default='copy', help='How to place files in flat dirs.')
    parser.add_argument('--dry_run', action='store_true', help='Only check and count pairs, do not write files.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f'Data root does not exist: {data_root}')

    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    if not splits:
        raise ValueError('No valid split provided in --splits')

    for split in splits:
        _flatten_split(data_root, split, args.mode, args.dry_run)


if __name__ == '__main__':
    main()

