# Fire-SmokeDataset

The dataset used to train and evaluate LHRF-YOLO, released with the paper
*LHRF-YOLO: A Lightweight Model with Hybrid Receptive Field for Forest Fire
Detection* (Forests 2025, 16, 1095).

**19,866 images**, split 6:3:1 into train / val / test.

| Split | Images |
|-------|--------|
| Train | 12,083 |
| Val | 5,640 |
| Test | 2,143 |
| **Total** | **19,866** |

The dataset was built by correcting the label quality of an existing fire and
smoke dataset: samples with quality problems were identified by image quality
assessment and re-annotated manually with LabelImg, then randomized into the
three splits to reduce sample bias.

## Classes

| Id | Name |
|----|------|
| 0 | `fire` |
| 1 | `smoke` |

## Access

The images are not redistributed in this repository. Download from one of the
mirrors below:

| Mirror | Link |
|--------|------|
| Baidu Netdisk | https://pan.baidu.com/s/1_Xti5AoIER3yZ5hQhToSZQ (code `m58i`) |
| Google Drive | https://drive.google.com/file/d/1VFYYXHbzDTtgjTUU8cl6qNgi8ZahxGSe/view?usp=sharing |
| QuarkDrive | https://pan.quark.cn/s/c13d3a6251c0 (code `4aii`) |

## Layout

`data_fire-smoke.yaml` expects the following structure, with `path` pointing at
the dataset root:

```text
FSDataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
│   └── labels/
```

## Label format

One `.txt` file per image, named after the image, in YOLO format: one line per
object with normalized `class x_center y_center width height`, values in `[0, 1]`
relative to the image width and height.

```text
# train/labels/000001.txt
0 0.512 0.481 0.142 0.223
1 0.734 0.556 0.310 0.281
```

The first field is the class index from the table above. A smoke plume is often
annotated as a `smoke` box that surrounds the visible smoke rather than a tight
mask, so boxes for the same fire event can overlap.

## Configuration

`data_fire-smoke.yaml` in the repository root:

```yaml
path: ../../../datasets/FSDataset
train:
 - train/images
val:
 - valid/images
test:
 - test/images
names:
 0: fire
 1: smoke
```

Change `path` to wherever you extracted the dataset. When training, Ultralytics
expects to find the matching labels under `images/../labels/`, which the layout
above already provides.

## License

Fire-SmokeDataset is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please cite the
original paper when reusing it.
