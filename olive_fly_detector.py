import os
import glob

import cv2
import numpy as np
from skimage.measure import label
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

_HERE = os.path.dirname(os.path.abspath(__file__))

POS_DIR = os.path.join(_HERE, "olive_fly")
NEG_DIR = os.path.join(_HERE, "not_olive_fly")
MODEL_PATH = os.path.join(_HERE, "olive_fly_model.joblib")

WORKING_SIZE = (160, 160)
HOG_CROP = (32, 32)
DECISION_THRESHOLD = 0.5

RANDOM_STATE = 42


def extract_foreground(img, kernel_size=9):
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
    largest_region = (labels == label_of_largest_region).astype(np.uint8)
    return largest_region


def extract_features(img):
    img = cv2.resize(img, WORKING_SIZE, interpolation=cv2.INTER_LINEAR)
    mask = extract_foreground(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = []
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        rect_area = w * h
        aspect = w / h if h else 0.0
        extent = area / rect_area if rect_area else 0.0
        solidity = area / hull_area if hull_area else 0.0
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter else 0.0
        hu = cv2.HuMoments(cv2.moments(c)).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

        f += [
            area / 25600.0, perimeter / 160.0, aspect, extent, solidity,
            circularity, w / 160.0, h / 160.0, int(mask.sum()) / 25600.0,
        ]
        f += list(hu)
        crop = cv2.resize(gray[y:y + h, x:x + w], HOG_CROP)
    else:
        f += [0.0] * 16
        crop = cv2.resize(gray, HOG_CROP)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = mask.astype(bool)
    if m.sum():
        for ch in range(3):
            vals = hsv[:, :, ch][m]
            f += [vals.mean() / 255.0, vals.std() / 255.0]
    else:
        f += [0.0] * 6

    f += list(hog(
        crop, orientations=8, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), feature_vector=True,
    ))
    return np.asarray(f, dtype=np.float32)


def _augment(img):
    return [
        img,
        img[:, ::-1],
        img[::-1, :],
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
    ]


def _list_images(folder):
    paths = []
    for ext in ("*.JPG", "*.jpg", "*.png", "*.PNG"):
        paths += glob.glob(os.path.join(folder, ext))
    return sorted(set(paths))


def _build(paths, label_value, augment=False):
    X, y = [], []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: could not read {path}")
            continue
        for variant in (_augment(img) if augment else [img]):
            X.append(extract_features(variant))
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

    Xtr_p, ytr_p = _build(pos_tr, 1, augment=True)
    Xtr_n, ytr_n = _build(neg_tr, 0, augment=False)
    Xte_p, yte_p = _build(pos_te, 1, augment=False)
    Xte_n, yte_n = _build(neg_te, 0, augment=False)

    Xtr = np.asarray(Xtr_p + Xtr_n, dtype=np.float32)
    ytr = np.asarray(ytr_p + ytr_n)
    Xte = np.asarray(Xte_p + Xte_n, dtype=np.float32)
    yte = np.asarray(yte_p + yte_n)

    print(f"Train: {Xtr.shape[0]} samples ({int(ytr.sum())} pos, "
          f"{Xtr.shape[1]} features), "
          f"Test: {Xte.shape[0]} samples ({int(yte.sum())} pos)")

    clf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    print(confusion_matrix(yte, pred))
    print(classification_report(
        yte, pred, target_names=["not_olive_fly", "olive_fly"], digits=3
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
        _model.n_jobs = 1
    return _model


def detect_olive_fly(image) -> bool:
    if image is None:
        return False
    vec = extract_features(image).reshape(1, -1)
    proba = _get_model().predict_proba(vec)[0, 1]
    return bool(proba >= DECISION_THRESHOLD)


if __name__ == "__main__":
    train()
