import requests

URL = "http://127.0.0.1:8000/predict"

# Данные для запроса
data = {
    "product_name": "Test Product",
    "ingredients": "Snail Secretion Filtrate, Betaine, Butylene Glycol,1,2-Hexanediol, Sodium Hyaluronate, Panthenol, Arginine, "
                   "Allantoin, Ethyl Hexanediol, Sodium Polyacrylate, Carbomer, Phenoxyethanol"
}

# POST-запрос к API
response = requests.post(URL, json=data)

# ответ
print("Response status:", response.status_code)
print("Response JSON:", response.json())
