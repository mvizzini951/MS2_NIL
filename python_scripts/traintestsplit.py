import pandas as pd
from sklearn.model_selection import train_test_split


def custom_train_test_split(df, valid_size_=0.15, test_size_=0.15, RANDOM_STATE=0):
    """
    :: Input(s) ::
        df - a dataframe containing all of the features and the target feature
        valid_size - the subset of the remaining training set after the test set was formed
        test_size - the size ofthe validation when compared to the training set
        RANDOM_STATE - a random state for this function to promote repeatability
    :: Output(s) ::
        X_train - training feature dataframe
        X_valid - validation feature dataframe
        X_test - testing feature dataframe
        y_train - training target feature
        y_valid - validation target feature
        y_test - testing target feature
    :: Function Description ::
        'custom_train_test_split' looks to take a cleaned dataframe and split it into the various training, validating, and testing datasets.
    """
    # Transform the continuous "SKILL_" column into a categorical such that it could be binned
    bins = [0, 80, 85, 90, 95, 100]
    labels = [
        "Very Low Skill",
        "Low Skill",
        "Medium Skill",
        "High Skill",
        "Very High Skill",
    ]
    df = df.copy()
    df["SKILL_CAT"] = pd.cut(df["SKILL"], bins=bins, labels=labels).astype(str)

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=RANDOM_STATE)

    # Get our X and y arrays
    # Target Feature: NILVAL_LONG_USD

    y = df.pop("NILVAL_LONG_USD")
    X = df

    # Split the data into testing and non-testing
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        train_size=(1 - valid_size_ - test_size_),
        stratify=X["SKILL_CAT"],
        random_state=RANDOM_STATE,
    )

    # Split the non-testing data into training and validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=valid_size_,
        stratify=X_train_valid["SKILL_CAT"],
        random_state=RANDOM_STATE,
    )

    return X_train, y_train, X_valid, y_valid, X_test, y_test


if __name__ == "__main__":
    print("traintestsplit .py file is still a WIP")
    print("Need to add argparse and other various pipeline functionality...")
