## Dataset

[Ames Iowa Housing Data](https://www.kaggle.com/datasets/marcopale/housing?select=AmesHousing.csv)

---

## Code

### Data preprocessing

1. Вивести перші 5 рядків (df.head())
2. Перевірити інформацію про датасет (df.info())
3. Обробити пропущені значення
   * або dropna()
   * або заповнення
4. Закодувати категоріальні ознаки
   * pd.get_dummies() або OneHotEncoder
5. Відокремити цільову змінну (SalePrice)
6. Розбити на train-test вибірки
7. Стандартизація (StandardScaler)

---

### Exploratory Data Analysis (EDA)

1. Матриця кореляцій (df.corr())
2. Теплова карта (heatmap)
3. Знайти ознаки з сильною кореляцією (multicollinearity)
4. VIF (Variance Inflation Factor)

---

### Baseline model (Linear Regression)

1. Навчити LinearRegression
2. Оцінити:
   * MAPE
   * MSE
   * R²
3. Переглянути коефіцієнти

---

### Multicollinearity analysis

* Знайти сильно скорельовані ознаки (>0.8)
* Показати:
  * що ознаки дублюють одна одну
  * чому це погано для лінійної регресії

---

### Ridge Regression

1. Навчити Ridge
2. Підібрати alpha (GridSearchCV)
3. Порівняти з Linear Regression

---

### Comparison of models

| Model             | MAPE | MSE | R² |
| ----------------- | ---- | --- | -- |
| Linear Regression |      |     |    |
| Ridge             |      |     |    |

---

### Conclusion

1. Ridge зменшує вплив мультиколінеарності
2. Оцінка точності моделі (MAPE, MSE, R²)
3. Висновки щодо обраних ознак
