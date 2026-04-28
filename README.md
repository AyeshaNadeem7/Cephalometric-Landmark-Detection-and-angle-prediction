---
title: Cephalometric Landmark & Angle Prediction
sdk: gradio
app_file: app.py
python_version: 3.10
---

Google Drive Link for dataset: https://drive.google.com/drive/folders/1_xJwVlx7u-gDJwzBZy4GNNqu7Bg7X-Bz

Project Objectives

•	Compute orthodontic angular measurements: SNA, SNB, and ANB angles.

•	Classify patients into skeletal malocclusion classes: Class I (Normal), Class II (Retruded Mandible), or Class III (Prognathic Mandible).

•	Provide a reusable inference pipeline that works on unseen patient X-ray images.

•	Enable model persistence through save/load of trained weights.

---

## Deploy on Hugging Face Spaces (Inference API)

This repo does **not** include a trained weights file by default.

1) Train the model using the notebook [orthodontis_angle_prediction.ipynb](orthodontis_angle_prediction.ipynb)

2) Save weights as `model.pth` (the notebook already contains `torch.save(model.state_dict(), "model.pth")`).

3) Create a new Hugging Face Space:
- **SDK**: Gradio
- **Python**: default

4) Upload/push these files to the Space repo:
- `app.py`
- `requirements.txt`
- `config.py`
- `model.pth`

5) After the Space builds, you will get a public URL like:
- `https://<username>-<space-name>.hf.space`

### Calling inference from your code

The Space exposes an API; the easiest stable way to call it is via `gradio_client`:

```python
from gradio_client import Client

client = Client("https://<username>-<space-name>.hf.space/")
result = client.predict("/path/to/xray.jpg", api_name="predict")
print(result)
```

In the Space UI, open **“Use via API”** to see the exact HTTP endpoint for your current Gradio version.

<img width="1500" height="744" alt="image" src="https://github.com/user-attachments/assets/e1344cc5-6e2c-43f1-a06a-9e0e8049e2e7" />
