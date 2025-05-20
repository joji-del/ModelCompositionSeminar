# ModelCompositionSeminar

Добро пожаловать в **ModelCompositionSeminar** — репозиторий, содержащий Jupyter Notebook (`ML-4.ipynb`), посвященный изучению композиций моделей машинного обучения, таких как **бэггинг** и **случайный лес**. Проект демонстрирует применение этих методов для задачи предсказания диабета у пациентов на основе датасета **Pima Indians Diabetes Database** с Kaggle.

## Обзор

Репозиторий включает Jupyter Notebook (`ML-4.ipynb`), который охватывает следующие темы:
- Загрузка и анализ датасета **Pima Indians Diabetes Database**.
- Разделение данных на обучающую и тестовую выборки.
- Применение моделей:
  - **Решающее дерево** (`DecisionTreeClassifier`).
  - **Бэггинг** (`BaggingClassifier`) на основе решающих деревьев.
  - **Случайный лес** (`RandomForestClassifier`).
- Оценка качества моделей с использованием метрик: **accuracy**, **precision**, **recall**, **ROC AUC**.
- Анализ важности признаков для случайного леса с визуализацией.
- Сравнение производительности моделей.

Ноутбук содержит подробные комментарии, код для визуализации данных и результатов, а также анализ важности признаков.

## Набор данных

- **Pima Indians Diabetes Database**:
  - Датасет содержит медицинские данные пациентов (768 записей) для предсказания наличия диабета (бинарная классификация: 0 — нет диабета, 1 — диабет).
  - Признаки (8):
    - `Pregnancies`: Количество беременностей.
    - `Glucose`: Уровень глюкозы в крови.
    - `BloodPressure`: Артериальное давление.
    - `SkinThickness`: Толщина кожной складки.
    - `Insulin`: Уровень инсулина.
    - `BMI`: Индекс массы тела.
    - `DiabetesPedigreeFunction`: Генетическая предрасположенность к диабету.
    - `Age`: Возраст.
  - Целевая переменная: `Outcome` (0 или 1).
  - Распределение классов:
    - 0 (нет диабета): 500 записей.
    - 1 (диабет): 268 записей.
  - Данные загружаются из файла `diabetes.csv`.

## Требования

Для работы с ноутбуком необходимы следующие библиотеки Python:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Установите их с помощью команды:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Структура репозитория

- `ML-4.ipynb`: Основной Jupyter Notebook с примерами бэггинга, случайного леса и анализа данных.
- `diabetes.csv`: Датасет для задачи предсказания диабета.
- `README.md`: Этот файл с описанием проекта.

## Использование

1. Склонируйте репозиторий:
   ```bash
   git clone https://github.com/<your-username>/ModelCompositionSeminar.git
   ```
2. Перейдите в папку репозитория:
   ```bash
   cd ModelCompositionSeminar
   ```
3. Убедитесь, что файл `diabetes.csv` находится в той же директории, что и ноутбук.
4. Запустите Jupyter Notebook:
   ```bash
   jupyter notebook ML-4.ipynb
   ```
5. Следуйте инструкциям в ноутбуке для выполнения анализа данных, обучения моделей и оценки результатов.

## Основные разделы ноутбука

1. **Загрузка и анализ данных**:
   - Чтение датасета с помощью `pd.read_csv('diabetes.csv')`.
   - Проверка структуры данных (`data.info()`): 768 записей, 9 столбцов (8 признаков + 1 целевая переменная), типы данных — `int64` (7 столбцов) и `float64` (2 столбца).
   - Просмотр первых строк (`data.head()`).
   - Анализ распределения целевой переменной (`data['Outcome'].value_counts()`): классы несбалансированы (500 — 0, 268 — 1).

2. **Подготовка данных**:
   - Разделение данных на признаки (`X`) и целевую переменную (`y`):
     ```python
     X = data.drop('Outcome', axis=1)
     y = data['Outcome']
     ```
   - Разделение на обучающую и тестовую выборки с помощью `train_test_split` (например, 80% — обучение, 20% — тест).

3. **Обучение моделей**:
   - **Решающее дерево**:
     ```python
     dt = DecisionTreeClassifier(random_state=42)
     dt.fit(X_train, y_train)
     y_pred = dt.predict(X_test)
     ```
   - **Бэггинг** (ансамбль решающих деревьев):
     ```python
     bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
     bagging.fit(X_train, y_train)
     y_pred = bagging.predict(X_test)
     ```
   - **Случайный лес**:
     ```python
     rf = RandomForestClassifier(n_estimators=100, random_state=42)
     rf.fit(X_train, y_train)
     y_pred = rf.predict(X_test)
     ```

4. **Оценка моделей**:
   - Используемые метрики:
     ```python
     accuracy = accuracy_score(y_test, y_pred)
     precision = precision_score(y_test, y_pred)
     recall = recall_score(y_test, y_pred)
     roc_auc = roc_auc_score(y_test, y_pred)
     ```
   - Сравнение производительности моделей (решающее дерево, бэггинг, случайный лес) по метрикам.

5. **Анализ важности признаков**:
   - Для случайного леса извлекаются важности признаков:
     ```python
     importances = rf.feature_importances_
     ```
   - Визуализация важности признаков с помощью горизонтальной столбчатой диаграммы:
     ```python
     indices = np.argsort(importances)
     columns = X.columns
     plt.figure()
     plt.title("Важность признаков")
     plt.barh(range(len(indices)), importances[indices], color='b', align='center')
     plt.yticks(range(len(indices)), columns[indices])
     plt.xlabel('Значимость признаков')
     plt.grid(True)
     plt.show()
     ```

## Пример использования

```python
# Загрузка данных
data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение случайного леса
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Оценка
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")
```


## Результаты

- **Решающее дерево**: Обычно показывает более низкую производительность из-за склонности к переобучению.
- **Бэггинг**: Улучшает стабильность и точность по сравнению с одним деревом за счет ансамблевого подхода.
- **Случайный лес**: Зачастую демонстрирует наилучшие результаты благодаря случайному выбору признаков и деревьев, что снижает переобучение и повышает обобщающую способность.
- Наиболее важные признаки (на основе случайного леса): `Glucose`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
