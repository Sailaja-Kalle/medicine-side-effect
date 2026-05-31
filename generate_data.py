import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

medicines = ["Paracetamol","Ibuprofen","Amoxicillin","Metformin","Aspirin",
             "Atorvastatin","Omeprazole","Cetirizine","Azithromycin","Lisinopril"]

medicine_side_effect_map = {
    "Paracetamol":   "Fatigue",
    "Ibuprofen":     "Stomach Pain",
    "Amoxicillin":   "Rash",
    "Metformin":     "Nausea",
    "Aspirin":       "Stomach Pain",
    "Atorvastatin":  "Fatigue",
    "Omeprazole":    "Dry Mouth",
    "Cetirizine":    "Drowsiness",
    "Azithromycin":  "Vomiting",
    "Lisinopril":    "Dizziness",
}

rows = []
for _ in range(3000):
    med      = random.choice(medicines)
    dosage   = random.choice(["Low","Medium","High"])
    age      = random.randint(18, 80)
    gender   = random.choice(["Male","Female"])
    duration = random.randint(1, 30)
    effect   = medicine_side_effect_map[med]
    rows.append([age, gender, med, dosage, duration, effect])

df = pd.DataFrame(rows, columns=["age","gender","medicine","dosage","duration_days","side_effect"])
df.to_csv("medicine_data.csv", index=False)
print("Dataset created! Shape:", df.shape)
print(df["side_effect"].value_counts())