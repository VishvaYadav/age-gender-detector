# Age & Gender Detector — Internship Extensions

A face-based **age, gender and ethnicity detector** trained on the [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new) dataset (my training project), extended with six internship tasks that reuse and build on the same pipeline.

## Base model
A multi-task CNN with one shared backbone and three output heads:
- **Gender:** male / female (sigmoid)
- **Ethnicity:** White, Black, Asian, Indian, Others (softmax)
- **Age:** years (regression)

Training notebook: [`notebooks/01_base_model_training.ipynb`](notebooks/01_base_model_training.ipynb)

## Internship tasks
| # | Task | Folder | Status |
|---|---|---|---|
| 1 | Car colour detection | [`tasks/task1_car_colour`](tasks/task1_car_colour) | ⏳ |
| 2 | Sign language detection | [`tasks/task2_sign_language`](tasks/task2_sign_language) | ⏳ |
| 3 | Nationality detection | [`tasks/task3_nationality`](tasks/task3_nationality) | ⏳ |
| 4 | Long hair identification | [`tasks/task4_long_hair`](tasks/task4_long_hair) | ⏳ |
| 5 | Age & emotion from voice | [`tasks/task5_voice_age`](tasks/task5_voice_age) | ⏳ |
| 6 | Senior citizen identification | [`tasks/task6_senior_citizen`](tasks/task6_senior_citizen) | ✅ |

## Project structure
```
age-gender-detector/
├── app.py            # Streamlit app: one page per task (streamlit run app.py)
├── app_pages/        # the app's pages
├── notebooks/        # training notebooks (run on Kaggle)
├── models/           # trained model files
├── utils/            # shared code: face detection + base-model prediction
│   ├── face_utils.py #   FaceDetector, AgeGenderPredictor, FaceAnalyzer, draw_results
│   └── demo.py       #   python -m utils.demo photo.jpg
├── tasks/            # one folder per internship task
└── docs/screenshots/ # screenshots for this README
```

## Setup
```bash
git clone <this-repo-url>
cd age-gender-detector
pip install -r requirements.txt
```

Run the app:
```bash
streamlit run app.py
```
It opens in your browser, with a page for the base model and one for each task.

Test the face pipeline on a single photo from the command line:
```bash
python -m utils.demo path/to/photo.jpg
```
It prints each face's predicted gender, age and ethnicity and saves an annotated copy next to the photo.

## How the pipeline works
1. **Face detection:** OpenCV's DNN face detector (res10 SSD) finds every face in the image.
2. **Cropping:** each box is enlarged by 25% so the crop matches UTKFace framing (forehead, chin, some hair).
3. **Preprocessing:** grayscale → 128×128 → scaled to 0–1, exactly as in training.
4. **Prediction:** the base model returns gender, ethnicity and age for all faces in one batch.
