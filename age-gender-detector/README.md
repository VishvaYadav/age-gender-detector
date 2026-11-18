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
| 6 | Senior citizen identification | [`tasks/task6_senior_citizen`](tasks/task6_senior_citizen) | ⏳ |

## Project structure
```
age-gender-detector/
├── notebooks/        # training notebooks (run on Kaggle)
├── models/           # trained model files
├── utils/            # shared code (face detection, prediction)
├── tasks/            # one folder per internship task
└── docs/screenshots/ # screenshots for this README
```

## Setup
```bash
git clone <this-repo-url>
cd age-gender-detector
pip install -r requirements.txt
```
