"""
从 Roboflow Universe 下载小型实例分割数据集 (YOLOv8-seg 格式)。
"""
from pathlib import Path
from roboflow import Roboflow

DATA_DIR = Path("data/instance_seg")
DATA_DIR.mkdir(parents=True, exist_ok=True)

rf = Roboflow()

DATASETS = [
    ("yolo-vtrf5",   "person-segmentation-in-room-ix55q", 1, "PersonRoom"),
    ("yoolov8seg",   "yolov8-seg-pv9im",                  1, "YOLOv8-seg-small"),
    ("yolov8-uph33", "seg-2g8uc",                         1, "Seg-General"),
]


def run():
    downloaded, failed = [], []
    for workspace, project, version, label in DATASETS:
        dest = DATA_DIR / label
        if dest.exists() and (dest / "data.yaml").exists():
            print(f"[OK] Already downloaded: {label}")
            downloaded.append(label)
            continue
        print(f"Downloading {label} ({workspace}/{project}/{version}) ...")
        try:
            proj = rf.workspace(workspace).project(project)
            ds = proj.version(version).download("yolov8-seg")
            print(f"  -> {ds.location}")
            downloaded.append(label)
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed.append((label, str(e)))

    print("\n" + "=" * 60 + "\nSummary\n" + "=" * 60)
    print(f"Downloaded: {downloaded}")
    print(f"Failed: {failed}")
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir():
            imgs = len(list(d.rglob("*.png"))) + len(list(d.rglob("*.jpg")))
            txts = len(list(d.rglob("*.txt")))
            print(f"  {d.name}/  images={imgs}  labels={txts}")


if __name__ == "__main__":
    run()
