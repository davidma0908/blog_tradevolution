def linear_model1():
    """
    線性回歸:正規方程
    :return:None
    """
    # 1.獲取數據
    data = load_boston()

    # 2.數據集劃分
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=22)

    # 3.特征工程-標準化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 4.機器學習-線性回歸(特征方程)
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 5.模型評估
    # 5.1 獲取系數等值
    y_predict = estimator.predict(x_test)
    print("預測值為:\n", y_predict)
    print("模型中的系數為:\n", estimator.coef_)
    print("模型中的偏置為:\n", estimator.intercept_)

    # 5.2 評價
    # 均方誤差
    error = mean_squared_error(y_test, y_predict)
    print("誤差為:\n", error)