## 📦 Requirements

Install dependencies using pip:

```bash
pip install torch torchvision einops tensorboard
✅ Optional: To enable GPU acceleration, ensure PyTorch is installed with CUDA support.

📊 Dataset Format
Each line in train.txt or val.txt should follow this format:

python-repl
image1.jpg 0
image2.jpg 1
...
Images should be placed in the corresponding folder (Images/train or Images/val).

The label should be an integer starting from 0.

🚀 How to Train
Run the script:

bash
python vit_train.py
Default Parameters
Input image shape: (3, 1, 256) → flattened to (3, 256)

Patch size: 16

Embedding dimension: 1024

Transformer depth: 6

Heads: 8

MLP hidden dim: 2048

Batch size: 64

Epochs: 10

Optimizer: Adam

Learning rate: 1e-4

The trained model will be saved to the model_save/ directory after each epoch.

📈 TensorBoard Logging
To monitor training and evaluation metrics:

bash
tensorboard --logdir=logs_vit_rewrite
Then open http://localhost:6006 in your browser.

🧠 Model Overview
This is a 1D-sequence version of the Vision Transformer (ViT)

Converts each image into a sequence of flattened patches

Applies standard Transformer layers with multi-head self-attention

Uses a classification token (cls_token) to summarize features

⚠️ Notes
Ensure that all image paths listed in train.txt and val.txt exist and are valid.

Number of classes is automatically inferred from dataset labels.

The training assumes normalized RGB images and resizes them to (1, 256).

📜 License
This project is licensed for educational and research purposes only. Feel free to modify or extend it for non-commercial uses.

❤️ Acknowledgments
ViT: An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)

PyTorch

einops
