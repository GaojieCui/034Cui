# 🔬 PyTorch Activation Function Visualization - Sigmoid

This project demonstrates how to visualize the effect of non-linear activation functions (specifically **Sigmoid**) on images using PyTorch and TensorBoard. It uses the CIFAR-10 dataset and logs the input/output image pairs before and after applying the Sigmoid activation.

---

## 📂 Project Structure

```

.
├── activation\_visualization.py  # Main script (this project)
├── sigmod\_logs/                 # TensorBoard logs will be saved here
└── dataset\_chen/                # CIFAR-10 dataset will be downloaded here

````

---

## 📦 Requirements

Install dependencies:

```bash
pip install torch torchvision tensorboard
````

> ✅ Recommended: Python 3.8+ and PyTorch 1.10+ with GPU support.

---

## 📊 Dataset

This script uses the **CIFAR-10 test set** (`train=False`) provided by `torchvision.datasets.CIFAR10`. It will be automatically downloaded to the specified path:

```python
root="D:\\intership\\DAY3\\dataset_chen"
```

---

## 🚀 How to Run

Simply execute the script:

```bash
python activation_visualization.py
```

This will:

1. Load CIFAR-10 test images.
2. Apply the **Sigmoid** activation function to the images.
3. Save both the original and processed images to TensorBoard logs.

---

## 📈 Visualize with TensorBoard

To view the input/output images:

```bash
tensorboard --logdir=sigmod_logs
```

Open [http://localhost:6006](http://localhost:6006) in your browser.

* `input`: Raw CIFAR-10 images
* `output`: Images after applying the Sigmoid activation

---

## 🧠 Model Overview

The model `Chen` is a simple neural network module with only an activation function:

```python
class Chen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        return self.sigmoid(input)
```

You can replace `Sigmoid` with `ReLU`, `Tanh`, or others to observe different effects.

---

## 📌 Notes

* The images are converted to tensors using `ToTensor()` which scales pixel values to `[0, 1]`.
* Applying `Sigmoid` again will further suppress the pixel intensity range.
* The output may look dimmer due to Sigmoid squashing effect.
