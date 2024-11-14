import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.missile_detector import MissileDetector
from utils.datasets import MissileDataset
from utils.transforms import Compose, Resize, ToTensor
import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = Compose([Resize((256, 256)), ToTensor()])
dataset = MissileDataset('data/', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = MissileDetector().to(device)
criterion_cls = nn.BCEWithLogitsLoss()
criterion_reg = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for batch in dataloader:
        images = batch['image'].to(device)
        bboxes = batch['bbox'].to(device)
        optimizer.zero_grad()
        outputs_cls, outputs_reg = model(images)
        loss_cls = criterion_cls(outputs_cls.squeeze(), torch.ones_like(outputs_cls.squeeze()))
        loss_reg = criterion_reg(outputs_reg.squeeze(), bboxes)
        loss = loss_cls + loss_reg
        loss.backward()
        optimizer.step()

model.eval()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, (256, 256))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        outputs_cls, outputs_reg = model(img_tensor)
    score = torch.sigmoid(outputs_cls).item()
    bbox = outputs_reg.squeeze().cpu().numpy()
    if score > 0.5:
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * frame.shape[1] / 256)
        y1 = int(y1 * frame.shape[0] / 256)
        x2 = int(x2 * frame.shape[1] / 256)
        y2 = int(y2 * frame.shape[0] / 256)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Missile Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
