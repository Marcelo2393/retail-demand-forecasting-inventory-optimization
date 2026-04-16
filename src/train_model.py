from lightgbm import LGBMRegressor

def train_model(X_train, y_train):
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        categorical_feature=["store_id", "item_id"]
    )

    return model