# CheckingCosmetics
Этот проект использует машинное обучение для предсказания комедогенности косметических продуктов на основе их состава. Модель анализирует список ингредиентов и предсказывает вероятность того, что продукт может вызвать образование комедонов.

## Структура проекта

- **train_model.py** - скрипт для обучения модели, включая загрузку данных, предварительную обработку, обучение и сохранение лучшей модели.
- **model_api.py** - приложение FastAPI для предоставления API, которое использует обученную модель для предсказания комедогенности на основе состава продукта.
- **inference.py** - скрипт для отправки запросов к API и получения предсказаний от модели.
- **data/cosmetics.csv** - исходные данные для обучения модели (информация о косметических продуктах).
- **best_cosmetic_model.pkl** - сохраненная обученная модель.

- ## Установка и запуск

1. Убедитесь, что у вас установлен Python 3.7+.
2. Клонируйте репозиторий и установите зависимости
3. Обучение модели:
Запустите скрипт train_model.py для обучения модели. Модель будет обучена, и лучший результат будет сохранен в файле best_cosmetic_model.pkl.
4. Запуск API:
Запустите скрипт model_api.py, чтобы создать сервер, который будет принимать запросы на предсказание
Сервер будет доступен по адресу http://127.0.0.1:8000.
6. Отправка запроса на предсказание:
Используйте скрипт inference.py для отправки запроса на сервер и получения предсказания
Пример запроса:
{
  "product_name": "Test Product",
  "ingredients": "Snail Secretion Filtrate, Betaine, Butylene Glycol, Sodium Hyaluronate, Panthenol"
}
Ответ будет содержать вероятность комедогенности:
{
  "product_name": "Test Product",
  "comedogenicity": 0.45
}
