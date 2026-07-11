from __future__ import annotations

FINAL_FEATURES = ["LLBCE", "PMBT", "MRT", "Depth.of.A"]
CONTINUOUS_CANDIDATES = [
    "LLBCE", "PMBT", "MRT", "Depth.of.A", "RAPL", "ART", "RH", "LSND", "age"
]
BINARY_CANDIDATES = ["sex_male", "jaw_deformity_type3", "third_molar_yes"]
CANDIDATE_FEATURES = CONTINUOUS_CANDIDATES + BINARY_CANDIDATES

TARGET_COL = "fracture_type"
PATIENT_ID_COL = "patient_id"
SIDE_COL = "side"
SAMPLE_COL = "sample"

COLUMN_ALIASES = {
    "fracture_type": [
        "fracture_type", "Type of fracture line", "Type.of.fracture.line",
        "type of fracture line", "fracture line type", "LSS"
    ],
    "patient_id": ["patient_id", "Patient ID", "patient", "case_id", "subject_id"],
    "sample": ["sample", "Sample", "sample_id", "side_id"],
    "side": ["side", "Side", "mandibular_side"],
    "Depth.of.A": [
        "Depth.of.A", "Depth of A", "Depth  of A", "Depth_of_A", "Depth.of. A", "Depth .of.A"
    ],
    "LLBCE": ["LLBCE", "LBCE", "Location of lateral bone cut end"],
    "PMBT": ["PMBT", "Posterior mandibular border thickness"],
    "MRT": ["MRT", "Mandibular ramus thickness"],
    "RAPL": ["RAPL", "Anteroposterior ramus length"],
    "ART": ["ART", "Anterior ramus thickness"],
    "RH": ["RH", "Ramus height"],
    "LSND": ["LSND", "Lingula sigmoid notch distance"],
    "age": ["age", "Age"],
    "sex": ["sex", "Sex", "gender"],
    "type of jaw deformity": [
        "type of jaw deformity", "type.of.jaw.deformity", "jaw deformity type", "skeletal_class"
    ],
    "third molar presence": [
        "third molar presence", "third.molar.presence", "third_molar_presence", "third molar"
    ],
}

MODEL_ORDER = [
    "Logistic Regression", "SVM", "Random Forest", "XGBoost", "CatBoost"
]

RANDOM_STATE_SPLIT = 70
RANDOM_STATE_MODEL = 42
N_CV_SPLITS = 5
TEST_FRACTION = 0.30
TOP_N_SUBSETS = 50
TOP50_THRESHOLD = 0.60
NEAR_OPTIMAL_RATIO = 0.90
