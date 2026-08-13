import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# based on https://github.com/kohpangwei/group_DRO/blob/master/dataset_scripts/generate_waterbirds.py

WATER_BIRDS = [
    "Albatross",
    "Auklet",
    "Cormorant",
    "Frigatebird",
    "Fulmar",
    "Gull",
    "Jaeger",
    "Kittiwake",
    "Pelican",
    "Puffin",
    "Tern",
    "Gadwall",
    "Grebe",
    "Mallard",
    "Merganser",
    "Guillemot",
    "Pacific_Loon",
]

PLACE_GROUPS = {
    0: ["bamboo_forest", "forest/broadleaf"],
    1: ["ocean", "lake/natural"],
}


def crop_and_resize(source_img, target_img):
    source_width, source_height = source_img.size
    target_width, target_height = target_img.size

    if (source_width < target_width) or (source_height < target_height):
        width_resize = (target_width, int((target_width / source_width) * source_height))
        if (width_resize[0] >= target_width) and (width_resize[1] >= target_height):
            source_resized = source_img.resize(width_resize, Image.LANCZOS)
        else:
            height_resize = (int((target_height / source_height) * source_width), target_height)
            source_resized = source_img.resize(height_resize, Image.LANCZOS)
        return crop_and_resize(source_resized, target_img)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height
    if source_aspect > target_aspect:
        new_source_width = int(target_aspect * source_height)
        offset = (source_width - new_source_width) // 2
        resize = (offset, 0, source_width - offset, source_height)
    else:
        new_source_height = int(source_width / target_aspect)
        offset = (source_height - new_source_height) // 2
        resize = (0, offset, source_width, source_height - offset)
    return source_img.crop(resize).resize((target_width, target_height), Image.LANCZOS)


def combine_and_mask(place, mask, bird_only):
    place_resized = crop_and_resize(place, bird_only)
    place_masked = np.around(np.asarray(place_resized) * (1 - mask)).astype(np.uint8)
    return Image.fromarray(np.asarray(bird_only) + place_masked)


def bird_label(img_filename):
    species = img_filename.split("/")[0].split(".")[1].lower()
    return int(any(name.lower() in species for name in WATER_BIRDS))


def load_cub(cub_dir):
    df = pd.read_csv(
        Path(cub_dir) / "images.txt",
        sep=" ",
        header=None,
        names=["img_id", "img_filename"],
        index_col="img_id",
    )
    df["y"] = [bird_label(path) for path in df["img_filename"]]
    return df.reset_index()


def place_files(places_dir, place_names):
    files = []
    for place_name in place_names:
        place_dir = Path(places_dir) / place_name[0] / place_name
        files.extend(
            f"/{place_name[0]}/{place_name}/{path.name}"
            for path in sorted(place_dir.glob("*.jpg"))
        )
    return files


def split_for_index(index, test_per_group, val_per_group):
    if index < test_per_group:
        return 2, "test"
    if index < test_per_group + val_per_group:
        return 1, "val"
    return 0, "train"


def make_composite(cub_dir, places_dir, row, place_filename):
    img_path = Path(cub_dir) / "images" / row["img_filename"]
    seg_path = Path(cub_dir) / "segmentations" / row["img_filename"].replace(".jpg", ".png")
    place_path = Path(places_dir) / place_filename[1:]

    img_np = np.asarray(Image.open(img_path).convert("RGB"))
    seg_gray = Image.open(seg_path).convert("L")
    seg_bin = (np.asarray(seg_gray, dtype=np.uint8) > 127).astype(np.uint8)
    seg_np = seg_bin[:, :, None]
    bird_only = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
    place = Image.open(place_path).convert("RGB")
    combined = combine_and_mask(place, seg_np, bird_only)
    mask = Image.fromarray((seg_bin * 255).astype(np.uint8), mode="L")
    return combined, mask, str(seg_path)


def run(args):
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    cub_dir = Path(args.cub_dir)
    places_dir = Path(args.places_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_cub(cub_dir)
    bird_pools = {
        y: df[df["y"] == y].sample(frac=1.0, random_state=args.seed + y).reset_index(drop=True)
        for y in (0, 1)
    }
    places = {place: place_files(places_dir, names) for place, names in PLACE_GROUPS.items()}
    for place, filenames in places.items():
        if not filenames:
            raise ValueError(f"No Places365 images found for place={place}")
        rng.shuffle(filenames)

    records = []
    for y in (0, 1):
        if len(bird_pools[y]) == 0:
            raise ValueError(f"No CUB examples found for y={y}")
        for place in (0, 1):
            group_dir = f"y{y}_place{place}"
            for n in tqdm(range(args.per_group), desc=f"y={y}, place={place}"):
                source = bird_pools[y].iloc[n % len(bird_pools[y])]
                place_filename = places[place][n % len(places[place])]
                split, split_name = split_for_index(n, args.test_per_group, args.val_per_group)
                source_stem = Path(source["img_filename"]).stem
                image_rel = Path("images") / group_dir / f"{source['img_id']}_{n:05d}_{source_stem}.jpg"
                mask_rel = Path("masks") / group_dir / f"{source['img_id']}_{n:05d}_{source_stem}.png"

                image_out = output_dir / image_rel
                mask_out = output_dir / mask_rel
                if not args.metadata_only:
                    image_out.parent.mkdir(parents=True, exist_ok=True)
                    mask_out.parent.mkdir(parents=True, exist_ok=True)
                    combined, mask, source_mask_path = make_composite(cub_dir, places_dir, source, place_filename)
                    combined.save(image_out)
                    mask.save(mask_out)
                else:
                    source_mask_path = str(Path(cub_dir) / "segmentations" / source["img_filename"].replace(".jpg", ".png"))

                records.append(
                    {
                        "img_filename": image_rel.as_posix(),
                        "mask_filename": mask_rel.as_posix(),
                        "source_img_id": int(source["img_id"]),
                        "source_img_filename": source["img_filename"],
                        "source_mask_path": source_mask_path,
                        "place_filename": place_filename,
                        "y": y,
                        "place": place,
                        "group_id": y * 2 + place,
                        "split": split,
                        "split_name": split_name,
                    }
                )

    meta = pd.DataFrame(records)
    meta.to_csv(output_dir / "metadata.csv", index=False)
    print(pd.crosstab(meta["y"], meta["place"]))
    print(pd.crosstab(meta["split_name"], [meta["y"], meta["place"]]))
    print(f"Wrote {len(meta)} rows to {output_dir / 'metadata.csv'}")

# create a larger waterbirds dataset
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub-dir")
    parser.add_argument("--places-dir")
    parser.add_argument(
        "--output-dir",
    )
    parser.add_argument("--per-group", type=int, default=6000)
    parser.add_argument("--test-per-group", type=int, default=250)
    parser.add_argument("--val-per-group", type=int, default=125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
