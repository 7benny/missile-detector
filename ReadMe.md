# 🚀 Missile Detection System

📁 **Structure**

```
missile_detector/
├── data/                🖼️
├── models/              🧠
│   ├── __init__.py
│   └── missile_detector.py
├── utils/               🛠️
│   ├── __init__.py
│   ├── datasets.py
│   └── transforms.py
├── main.py              🎯
├── requirements.txt     📋
└── README.md            📖
```

---

🔧 **Installation**

1. **Install dependencies** 📥:

   ```bash
   pip install -r requirements.txt
   ```

2. **Place data** 📂: Put images and annotations in `data/` directory.

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

---

📦 **Dependencies**

- `torch`
- `torchvision`
- `opencv-python`
- `numpy`

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

👍 **Usage**

- **Start training** and **real-time detection** with one command 🎉
