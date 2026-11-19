# models/

| File | Made by | What it does |
|---|---|---|
| `base_age_gender_ethnicity.keras` | `notebooks/01_base_model_training.ipynb` | predicts age, gender and ethnicity from a face |
| `labels.json` | same notebook | label names and input size for the base model |
| `face_detector/deploy.prototxt` | [OpenCV samples](https://github.com/opencv/opencv/tree/4.x/samples/dnn/face_detector) | network definition of OpenCV's res10 SSD face detector |
| `face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel` | [OpenCV 3rd-party files](https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20180205_fp16) | pretrained weights for that face detector |

The two base-model files come from the Kaggle notebook's **Output** tab (`/kaggle/working/models/`).
