# CS336 Assignment 1: Basics

Welcome to my solution for **CS336 Assignment 1: Basics**! This repository contains a comprehensive implementation of foundational concepts in deep learning and natural language processing (NLP). Each file is carefully designed to address specific aspects of the assignment, with a focus on clarity, modularity, and extensibility.

---

## 🌟 **Key Features of My Solution**
- **Well-structured Codebase**: Each module is self-contained and focuses on a specific functionality, making the code easy to understand and extend.
- **Custom Implementations**: Core components like tokenizers, optimizers, and attention mechanisms are implemented from scratch, showcasing a deep understanding of the underlying principles.
- **Extensive Documentation**: Each file and function is well-documented, ensuring clarity for anyone reviewing the code.
- **Reproducibility**: The code is designed to be easily reproducible, with clear instructions for running experiments and training models.

---

## 📂 **Directory Structure and File Descriptions**

### **Core Modules**
1. **`__init__.py`**
   - Marks the directory as a Python package. Ensures all modules can be imported seamlessly.

2. **`adamw.py`**
   - Implements the **AdamW optimizer**, a variant of Adam with weight decay regularization. This is a critical component for training modern deep learning models.

3. **`bpe_tokenizer.py`**
   - Implements a **Byte Pair Encoding (BPE) tokenizer** from scratch. This tokenizer is widely used in NLP tasks to handle subword tokenization efficiently.

4. **`bpe_train.py`**
   - Provides utilities to train the BPE tokenizer on a given dataset. Includes methods for vocabulary generation and subword merging.

5. **`checkpointing.py`**
   - Handles **model checkpointing** during training. Ensures that training progress can be saved and resumed seamlessly.

6. **`cross_entropy.py`**
   - Implements the **cross-entropy loss function**, a fundamental loss function for classification tasks.

7. **`decode.py`**
   - Contains utilities for decoding tokenized sequences back into human-readable text.

8. **`embedding.py`**
   - Implements **word embeddings**, mapping tokens to dense vector representations. This is a foundational component for NLP models.

9. **`get_batch.py`**
   - Provides utilities for batching data during training, ensuring efficient data loading and processing.

10. **`gradient_clipping.py`**
    - Implements **gradient clipping**, a technique to prevent exploding gradients during backpropagation.

11. **`linear.py`**
    - Implements a **linear layer** (fully connected layer), a building block for neural networks.

12. **`lr_cosine_schedule.py`**
    - Implements a **cosine learning rate scheduler**, which adjusts the learning rate during training for better convergence.

13. **`multihead_self_attention.py`**
    - Implements **multi-head self-attention**, a core component of the Transformer architecture.

14. **`omg.py`**
    - A utility file for miscellaneous functions or experiments related to the assignment.

15. **`pretokenization_example.py`**
    - Demonstrates **pre-tokenization** techniques, preparing raw text for tokenization.

16. **`prompt.txt`**
    - Contains example prompts or input data for testing the tokenizer and model.

17. **`rmsnorm.py`**
    - Implements **RMSNorm**, a normalization technique used in some Transformer variants.

18. **`rope.py`**
    - Implements **Rotary Positional Embeddings (RoPE)**, a technique for encoding positional information in Transformers.

19. **`scaled_dot_product_attention.py`**
    - Implements **scaled dot-product attention**, the core operation in self-attention mechanisms.

20. **`softmax.py`**
    - Implements the **softmax function**, used for converting logits into probabilities.

21. **`swiglu.py`**
    - Implements the **SwiGLU activation function**, a variant of the GLU activation used in some Transformer models.

22. **`training_togerther.py`**
    - A script for training the model, integrating all components like tokenization, embedding, attention, and optimization.

23. **`transformer_block.py`**
    - Implements a **Transformer block**, combining multi-head self-attention, feedforward layers, and normalization.

24. **`transformer_lm.py`**
    - Implements a **Transformer-based language model**, integrating all components to build a complete NLP model.

---
