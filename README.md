# Computer Vision: Image Classification

Image classification projects built with TensorFlow / Keras. Each project lives in its own folder with its data-preparation script and training notebook.

| Project | Task | Model | Validation accuracy |
|---|---|---|---|
| [Cats vs Dogs](#cats-vs-dogs) | Binary classification (cat / dog) | CNN from scratch | 80.7% |

---

## Cats vs Dogs

A convolutional neural network trained from scratch to tell cats from dogs. It is evaluated on accuracy, precision, recall, F1-score and AUC.

### Dataset

[Kaggle — Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats)

| Split | Images | Labels |
|---|---|---|
| `train/` | 25,000 (12,500 cats, 12,500 dogs) | Yes, from the filename (`cat.0.jpg`, `dog.0.jpg`) |
| `test1/` | 12,500 | No (Kaggle kept them private) |

Because `test1/` has no labels, 30% of `train/` is held out as a labelled validation set:

| Subset | Images |
|---|---|
| Training | 17,500 |
| Validation | 7,500 |

The split uses a fixed seed (`SEED = 42`), so it is reproducible and the two subsets never overlap.

### Project structure

```
cv_image_classification/
├── README.md
├── .gitignore
└── cat_vs_dog/
    ├── clean_prepare_data.py   # sorts train images into cats/ and dogs/
    ├── training.ipynb          # data loading, model, training, evaluation
    └── dogs-vs-cats/           # dataset (not tracked by git)
        ├── train/
        │   ├── cats/
        │   └── dogs/
        ├── test1/
        └── sampleSubmission.csv
```

### Setup

**1. Clone the repository**

```bash
git clone https://github.com/BodeMurairi2/cv_image_classification.git
cd cv_image_classification
```

**2. Install the dependencies**

```bash
pip install tensorflow keras numpy matplotlib jupyter
```

Tested with Python 3.13, TensorFlow 2.21, Keras 3.15, NumPy 2.1 and Matplotlib 3.10. A GPU is optional: the results below were obtained on CPU, at roughly 2 minutes per epoch.

**3. Download and extract the data**

Download `dogs-vs-cats.zip` from the [Kaggle competition page](https://www.kaggle.com/c/dogs-vs-cats/data) (you need a Kaggle account and must accept the competition rules). Then:

```bash
cd cat_vs_dog
unzip dogs-vs-cats.zip -d dogs-vs-cats
cd dogs-vs-cats
unzip train.zip && unzip test1.zip
cd ..
```

**4. Sort the training images into class folders**

```bash
python clean_prepare_data.py
```

This moves every image in `train/` into `train/cats/` or `train/dogs/` based on its filename. Keras uses this folder layout to assign labels (cats → `0`, dogs → `1`).

**5. Train**

Open `cat_vs_dog/training.ipynb` and run all cells. The notebook builds paths from the current working directory, so run it from inside `cat_vs_dog/` (Jupyter and VS Code do this by default).

### Pipeline

1. **Explore:** count the images per class and show a 4×4 grid of random cats and dogs.
2. **Load:** `keras.utils.image_dataset_from_directory` resizes images to 200×200 and batches them (batch size 32). It creates the 70/30 training/validation split, and a separate unlabelled test set with `labels=None, shuffle=False` so predictions stay in file order.
3. **Input pipeline:** the training data is shuffled (buffer 1,000) and both sets use `prefetch(AUTOTUNE)`. Data is not cached in memory: as float32 at 200×200, the full dataset would need about 12 GB of RAM.
4. **Model:** see below.
5. **Train:** RMSprop (learning rate 1e-4), binary cross-entropy loss, 10 epochs.

### Model

| Layer | Output shape | Parameters |
|---|---|---|
| Rescaling (1/255) | 200 × 200 × 3 | 0 |
| Conv2D 16, 3×3, ReLU + MaxPool 2×2 | 99 × 99 × 16 | 448 |
| Conv2D 32, 3×3, ReLU + MaxPool 2×2 | 48 × 48 × 32 | 4,640 |
| Conv2D 32, 3×3, ReLU + MaxPool 2×2 | 23 × 23 × 32 | 9,248 |
| Conv2D 64, 3×3, ReLU + MaxPool 2×2 | 10 × 10 × 64 | 18,496 |
| Flatten | 6,400 | 0 |
| Dense 64, ReLU (×3) | 64 | 417,984 |
| Dense 1, sigmoid | 1 | 65 |
| **Total** | | **450,881** |

### Results

Validation metrics after 10 epochs (7,500 images). The positive class is **dog**, and the threshold is 0.5.

| Metric | Training | Validation |
|---|---|---|
| Accuracy | 0.825 | **0.807** |
| Precision | 0.828 | 0.857 |
| Recall | 0.818 | 0.742 |
| F1-score | 0.823 | 0.795 |
| AUC | 0.908 | 0.895 |
| Loss | 0.386 | 0.423 |

**What this shows**

- Training and validation scores are close, so the model is not overfitting yet. Validation loss was still falling at epoch 10, so more epochs would likely help.
- Precision is higher than recall: when the model says "dog" it is usually right, but it misses about a quarter of the dogs, labelling them as cats.
- Validation precision and recall swing a lot from epoch to epoch (for example, recall was 0.93 at epoch 4 and 0.61 at epoch 5). The decision boundary is not stable yet.

### Next steps

- [ ] Plot the training and validation curves from `history`
- [ ] Train for more epochs, with `EarlyStopping` on `val_loss`
- [ ] Add data augmentation (random flip, rotation, zoom) to improve generalisation
- [ ] Add `Dropout` before the dense layers
- [ ] Evaluate with a confusion matrix and a classification report
- [ ] Predict on `test1/` and write a Kaggle submission (`id,label`, where `label` is the probability of dog)
- [ ] Compare with transfer learning (e.g. MobileNetV2 or EfficientNet pretrained on ImageNet)

### Notes

- The dataset, zip archives, model weights and logs are excluded by `.gitignore`, so the repository holds code only.
- `image_dataset_from_directory` sorts filenames as text (`1.jpg, 10.jpg, 100.jpg, …`). Use `test_data.file_paths` to match each prediction to its image ID.

## Author

**Bode Murairi** — [GitHub](https://github.com/BodeMurairi2)
