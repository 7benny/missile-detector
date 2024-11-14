# 🚀 Missile Detection System

📁 **Project Structure**

```
missile_detection/
├── data/                🖼️
├── models/              🧠
│   ├── __init__.py
│   └── missile_detector.py
├── utils/               🛠️
│   ├── __init__.py
│   ├── datasets.py
│   └── transforms.py
├── cython/              ⚙️   # Uses OpenMP
│   ├── __init__.py
│   └── processing.pyx
├── main.py              🎯
├── setup.py             🔧
├── requirements.txt     📋
└── README.md            📖
```

---

🔧 **Installation**

1. **Install dependencies** 📥:

   ```bash
   pip install -r requirements.txt
   ```

2. **Build Cython extension** 🛠️:

   ```bash
   python setup.py build_ext --inplace
   ```

3. **Place data** 📂: Put images and annotations in the `data/` directory.

---

▶️ **Run**

- **Train and detect** 🎯:

  ```bash
  python main.py
  ```

---

🧠 **Components**

- **Model** (`models/missile_detector.py`) 🧠
- **Dataset loader** (`utils/datasets.py`) 📄
- **Transforms** (`utils/transforms.py`) 🔄
- **Cython Processing** (`cython/processing.pyx`) ⚡ (Uses OpenMP)

---

📦 **Dependencies**

- `torch`
- `torchvision`
- `opencv-python`
- `numpy`
- `cython`

---

📸 **Detection**

- **Webcam feed** 📹
- **Displays bounding boxes** 🔲

---

⚙️ **Configuration**

- **Adjust parameters** in `main.py` ⚙️
- **Modify model** in `models/missile_detector.py` 📝

---

📊 **Data Format**

- **Images**: `.jpg`, `.png` 🖼️
- **Annotations**: `.txt` files with bounding boxes 📝

---

💡 **OpenMP Usage**

- **OpenMP** is utilized in `cython/processing.pyx` via Cython's `cython.parallel` module.
- **Compilation flags** in `setup.py` include `-fopenmp` to enable OpenMP support.
- **No explicit import** of OpenMP is needed in the code; it's handled during compilation.

---