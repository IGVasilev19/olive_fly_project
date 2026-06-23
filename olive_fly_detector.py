import os
import glob

import cv2
import numpy as np
from skimage.measure import label
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

_HERE = os.path.dirname(os.path.abspath(__file__))

POS_DIR = os.path.join(_HERE, "olive_fly")
NEG_DIR = os.path.join(_HERE, "not_olive_fly")
MODEL_PATH = os.path.join(_HERE, "olive_fly_model.joblib")

WORKING_SIZE = (160, 160)
MODEL_SIZE = (80, 80)

RANDOM_STATE = 42


def extract_foreground(img, kernel_size=9, background_color=255):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    _, img_bw = cv2.threshold(
        img_gray, -1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((kernel_size, kernel_size))
    img_bw_cleaned = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel)

    labels = label(img_bw_cleaned)
    label_of_largest_region = np.argmax(
        np.bincount(labels.flat, weights=img_bw_cleaned.flat)
    )
    largest_region = labels == label_of_largest_region

    x, y = np.where(np.invert(largest_region))
    foreground = img.copy()
    foreground[x, y] = background_color

    return foreground, largest_region


def _foreground_image(img):
    img = cv2.resize(img, WORKING_SIZE, interpolation=cv2.INTER_LINEAR)
    fg, _ = extract_foreground(img)
    fg = cv2.resize(fg, MODEL_SIZE, interpolation=cv2.INTER_AREA)
    fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return fg


def preprocess(img):
    return _foreground_image(img).flatten()


def _augment(fg):
    return [
        fg,
        fg[:, ::-1],
        fg[::-1, :],
        np.rot90(fg, 1),
        np.rot90(fg, 2),
        np.rot90(fg, 3),
    ]


def _list_images(folder):
    paths = []
    for ext in ("*.JPG", "*.jpg", "*.png", "*.PNG"):
        paths += glob.glob(os.path.join(folder, ext))
    return sorted(set(paths))


def _features(paths, label_value, augment=False):
    X, y = [], []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: could not read {path}")
            continue
        fg = _foreground_image(img)
        variants = _augment(fg) if augment else [fg]
        for v in variants:
            X.append(v.flatten())
            y.append(label_value)
    return X, y


def train():
    pos = _list_images(POS_DIR)
    neg = _list_images(NEG_DIR)
    if not pos or not neg:
        raise SystemExit(
            f"Need images in both {POS_DIR} and {NEG_DIR} "
            f"(found {len(pos)} pos, {len(neg)} neg)."
        )
    print(f"Found {len(pos)} olive-fly, {len(neg)} non-olive-fly images")

    pos_tr, pos_te = train_test_split(
        pos, test_size=0.2, random_state=RANDOM_STATE
    )
    neg_tr, neg_te = train_test_split(
        neg, test_size=0.2, random_state=RANDOM_STATE
    )

    Xtr_p, ytr_p = _features(pos_tr, 1, augment=True)
    Xtr_n, ytr_n = _features(neg_tr, 0, augment=False)
    Xte_p, yte_p = _features(pos_te, 1, augment=False)
    Xte_n, yte_n = _features(neg_te, 0, augment=False)

    Xtr = np.asarray(Xtr_p + Xtr_n, dtype=np.float32)
    ytr = np.asarray(ytr_p + ytr_n)
    Xte = np.asarray(Xte_p + Xte_n, dtype=np.float32)
    yte = np.asarray(yte_p + yte_n)

    print(f"Train: {Xtr.shape[0]} samples ({ytr.sum()} pos, "
          f"{Xtr.shape[1]} features), "
          f"Test: {Xte.shape[0]} samples ({yte.sum()} pos)")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )
    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte)
    print(confusion_matrix(yte, pred))
    print(classification_report(
        yte, pred, target_names=["not_olive_fly", "olive_fly"]
    ))

    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    return clf


_model = None


def _get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run "
                f"`python olive_fly_detector.py` to train it first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def detect_olive_fly(image) -> bool:
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    if img is None:
        return False

    vec = preprocess(img).reshape(1, -1)
    return bool(_get_model().predict(vec)[0] == 1)


if __name__ == "__main__":
    train()
