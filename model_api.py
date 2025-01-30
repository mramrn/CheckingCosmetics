import pandas as pd
from fastapi import FastAPI, Form
import joblib
from pydantic import BaseModel
from typing import List

#список всех возможных ингредиентов
ALL_INGREDIENTS = [
    "glycerin", "sodiumhyaluronate", "centellaasiaticaextract", "squalane",
    "gluconolactone", "sodiumlactate", "salicylicacid", "niacinamide", "ceramide",
    "chamomillarecutitaflowerextract", "camelliasinensisleafextract",
    "glycyrrhizaglabrarootextract", "betaine", "polygonumcuspidatumrooextract",
    "allantoin", "butyleneglycol", "titaniumdioxide", "alcoholdenat", "carbomer",
    "sodiumpolyacrylate", "caprylylglycol", "ethylhexylglycerin", "octocrylene",
    "coco_caprylatecaprate", "cetearylalcohol", "cocosnuciferaoil", "retinol",
    "hydrolyzedcollagen", "madecassoside", "tocopherol", "phyticacid",
    "maltodextrin", "dipropyleneglycol", "tromethamine", "biosaccharidgum1",
    "caffeine", "rosacaninafruitextract", "menthol", "peg60hydrogenatedcastoroil",
    "dimethicone", "rosmarinusofficinalisleafoil", "triacetin", "dexpanthenol",
    "zincpca", "cynarascolymusleafextract", "cholesterol", "peptides",
    "snailsecretionfiltrate", "ascorbicacid"
]


# Словарь синонимов
INGREDIENT_SYNONYMS = {
    "ceramide 1": "ceramide", "ceramide 2": "ceramide", "ceramide 3": "ceramide",
    "ceramide 4": "ceramide", "ceramide 5": "ceramide", "ceramide 6-II": "ceramide",
    "ceramide 7": "ceramide", "ceramide 8": "ceramide", "ceramide 9": "ceramide",
    "ceramide np": "ceramide", "ceramide ap": "ceramide", "ceramide eop": "ceramide",
    "ceramide eos": "ceramide", "ceramide ng": "ceramide",
    "retinol": "retinoid", "retinoids": "retinoid", "retinoid esters": "retinoid",
    "ascorbic acid": "ascorbicacid", "vitamin c": "ascorbicacid", "vitamin b3": "niacinamide",
    "hyaluronic acid": "hyaluronicacid", "sodium hyaluronate": "hyaluronicacid", "snail mucin": "snailsecretionfiltrate",
    "peptides": "peptides", "green tea extract": "camelliasinensisleafextract",
    "coconut oil": "cocosnuciferaoil", "lanolin": "lanolin", "avocado oil": "perseagratissimaoil",
    "palm oil": "palmoil", "shea butter": "sheabutter", "dimethicone": "dimethicone",
    "silicone": "dimethicone", "cyclopentasiloxane": "dimethicone", "polydimethylsiloxane": "dimethicone",
    "dimethylpolysiloxane": "dimethicone", "dimethylsilicone": "dimethicone"
}



# Функция нормализации ингредиентов
def normalize_ingredients(ingredients: List[str]) -> List[str]:
    """Приводит ингредиенты к единому формату, заменяя синонимы."""
    normalized = set()
    for ing in ingredients:
        ing_lower = ing.lower().strip()  # Приводит к нижнему регистру и убирает лишние пробелы
        if ing_lower in INGREDIENT_SYNONYMS:
            normalized.add(INGREDIENT_SYNONYMS[ing_lower])
        else:
            normalized.add(ing_lower)
    return list(normalized)

app = FastAPI()

with open('best_cosmetic_model.pkl', 'rb') as file:
    model = joblib.load(file)

# модель данных для входа
class ProductComposition(BaseModel):
    product_name: str
    ingredients: str

# модель данных для выхода
class Prediction(BaseModel):
    product_name: str
    comedogenicity: float

# Проверка статуса сервера
@app.get('/status')
def status():
    return {"status": "I'm OK"}

# Версия модели
@app.get('/version')
def version():
    return {"model_version": "1.2"}

# Эндпоинт для предсказания
@app.post('/predict', response_model=Prediction)
def predict(data: ProductComposition):
    ingredient_list = [ingredient.strip() for ingredient in data.ingredients.split(",")]

    # Нормализация ингредиентов (учет синонимов)
    normalized_ingredients = normalize_ingredients(ingredient_list)

    # Создание вектора признаков (0 и 1)
    input_data = {ingredient: (1 if ingredient in normalized_ingredients else 0) for ingredient in ALL_INGREDIENTS}

    # Преобразование в DataFrame
    input_df = pd.DataFrame([input_data])

    # Предсказание вероятности
    pred_prob = model.predict_proba(input_df)[:, 1]

    return Prediction(
        product_name=data.product_name,
        comedogenicity=float(pred_prob[0])
    )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run("model_api:app", host="0.0.0.0", port=8000, reload=True, access_log=False)

