"""Train and compare three models on the SEER breast cancer data.

Phase 2: plain training and evaluation. Phase 3 adds MLflow tracking.

Run from the project root with:  python -m src.train
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import pandas as pd  # tables: used here to print the results neatly

# ColumnTransformer: applies a different preprocessing step to each group of columns.
from sklearn.compose import ColumnTransformer

# The two tree-based models we compare against the linear baseline.
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

# The simple baseline model. If fancy models can't beat it, they aren't worth it.
from sklearn.linear_model import LogisticRegression

# Metrics. All of them treat Dead (= 1) as the "positive" class.
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Splits the data into a training part and a held-out test part.
from sklearn.model_selection import train_test_split

# Pipeline: chains preprocessing + model into one object.
from sklearn.pipeline import Pipeline

# Preprocessing tools:
#   OneHotEncoder  -> unordered categories (Race) become 0/1 columns
#   OrdinalEncoder -> ordered categories (T1 < T2 < T3 < T4) become 0, 1, 2, 3
#   StandardScaler -> numbers rescaled to mean 0, std 1
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Our own module: the feature lists and the cleaned data.
# Keeping these in one place (src/data.py) means training, the API and the tests
# all use exactly the same columns and cleaning rules.
from src.data import NOMINAL_FEATURES, NUMERIC_FEATURES, ORDINAL_FEATURES, load_data

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# Fixed random seed: the split and the models come out identical on every run,
# so results can be reproduced and compared fairly.
RANDOM_STATE = 42

# 20% of patients are held back for testing; the model never sees them in training.
TEST_SIZE = 0.2


def build_preprocessor() -> ColumnTransformer:
    """Apply the right transformation to each group of columns."""
    return ColumnTransformer(
        transformers=[
            # (name, transformer, columns)

            # Numeric columns: rescale so large numbers (Tumor Size up to 140)
            # don't dominate small ones (nodes positive). Logistic Regression
            # needs this; tree models ignore scale, so it doesn't hurt them.
            ("num", StandardScaler(), NUMERIC_FEATURES),

            # Ordered categories: we pass the order explicitly
            # (e.g. ["T1", "T2", "T3", "T4"] -> 0, 1, 2, 3) so the numbers follow
            # the medical severity. Without it, sklearn would sort alphabetically.
            (
                "ord",
                OrdinalEncoder(categories=list(ORDINAL_FEATURES.values())),
                list(ORDINAL_FEATURES),
            ),

            # Unordered categories: one 0/1 column per value (Race_Black,
            # Race_White, ...), so no fake order like Black < Other < White.
            # handle_unknown="ignore": an unseen value in the API becomes all
            # zeros instead of crashing the prediction.
            ("nom", OneHotEncoder(handle_unknown="ignore"), NOMINAL_FEATURES),
        ]
    )


def get_models() -> dict:
    """The three candidate models, keyed by name.

    class_weight="balanced" makes a mistake on the rare Dead class (15%) count
    about 5.5x more than a mistake on Alive. Without it, a model can get 85%
    accuracy by predicting "Alive" for everyone, which means 0% recall on Dead.
    """
    return {
        # Baseline: simple, fast, explainable (each feature gets a weight).
        # max_iter=1000: gives the solver enough steps to converge.
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),

        # Many decision trees, each trained on a random sample; they vote.
        "random_forest": RandomForestClassifier(
            n_estimators=300,      # number of trees: more = more stable, slower
            min_samples_leaf=5,    # each leaf needs >= 5 patients: limits overfitting
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,             # use all CPU cores
        ),

        # Gradient boosting: trees built one after another, each fixing the
        # previous ones' errors. The "Hist" version is sklearn's fast
        # implementation and, unlike GradientBoostingClassifier, accepts class_weight.
        "gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


def build_pipeline(model) -> Pipeline:
    """Preprocessing + model as one object.

    Why a Pipeline:
    1. No leakage: the scaler and encoders learn their settings (means,
       categories) inside fit(), from the training data only.
    2. No training/serving skew: the saved model includes its preprocessing,
       so the API passes raw values and they're transformed exactly as in training.
    """
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Score a trained pipeline on the held-out test set."""
    # Hard predictions: 0 (Alive) or 1 (Dead), using the default 0.5 cut-off.
    y_pred = pipeline.predict(X_test)

    # Probability of Dead for each patient. predict_proba returns two columns
    # [P(Alive), P(Dead)]; [:, 1] takes the Dead column.
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        # Of all patients who actually died, how many did we flag?
        # Our main metric: a missed high-risk patient is the costliest error.
        "recall": recall_score(y_test, y_pred),

        # Of all patients we flagged as Dead, how many actually died?
        # Low precision = many false alarms. Guards against "flag everyone".
        "precision": precision_score(y_test, y_pred),

        # Harmonic mean of precision and recall: one number balancing both.
        "f1": f1_score(y_test, y_pred),

        # How well the model ranks patients by risk, ignoring the cut-off.
        # 0.5 = random guessing, 1.0 = perfect ranking. Uses probabilities.
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main() -> None:
    # 1. Load the cleaned data: X = 13 features, y = 1 (Dead) / 0 (Alive).
    X, y = load_data()

    # 2. Split into train (80%) and test (20%).
    #    stratify=y keeps the 15.3% Dead rate the same in both parts; without it,
    #    a random split could leave too few Dead patients in the test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, "
          f"Dead rate train/test: {y_train.mean():.3f}/{y_test.mean():.3f}")

    # 3. Train and score each model. Each gets a fresh pipeline, so they
    #    never share fitted preprocessing.
    results = {}
    for name, model in get_models().items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)           # learns ONLY from training data
        results[name] = evaluate(pipeline, X_test, y_test)  # scored on unseen data

    # 4. Show a comparison table: one row per model, best recall first.
    #    .T flips the table so models are rows and metrics are columns.
    table = pd.DataFrame(results).T.round(3).sort_values("recall", ascending=False)
    print(table)


# Runs only when this file is executed directly (python -m src.train),
# not when another file imports it (e.g. tests importing build_pipeline).
if __name__ == "__main__":
    main()