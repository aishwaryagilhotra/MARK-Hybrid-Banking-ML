print("\n===== REGRESSION RESULTS =====")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in reg_models.items():

    cv_rmse = -cross_val_score(
        model, Xr, yr,
        cv=kf,
        scoring='neg_root_mean_squared_error'
    )
    print(f"\n{name} CV RMSE:", cv_rmse.mean())

    model.fit(X_train_r, y_train_r)
    preds = model.predict(X_test_r)

    print("RMSE:", np.sqrt(mean_squared_error(y_test_r, preds)))
    print("MAE:", mean_absolute_error(y_test_r, preds))
    print("R2:", r2_score(y_test_r, preds))

    # Predicted vs Actual
    plt.figure()
    plt.scatter(y_test_r, preds)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name} Predicted vs Actual")
    plt.show()

    # Residual Plot
    residuals = y_test_r - preds
    plt.figure()
    plt.scatter(preds, residuals)
    plt.axhline(0, linestyle='--')
    plt.title(f"{name} Residual Plot")
    plt.show()
