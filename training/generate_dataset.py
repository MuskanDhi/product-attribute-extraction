import pandas as pd
import random
import os

random.seed(42)

categories = [
    "Wedding Dress",
    "Bridesmaid Dress",
    "Prom Gown",
    "Evening Gown",
    "Cocktail Dress"
]

silhouettes = [
    "A-Line",
    "Ball Gown",
    "Mermaid",
    "Sheath",
    "Empire",
    "Fit and Flare",
    "Trumpet",
    "Straight"
]

fabrics = [
    "Chiffon",
    "Satin",
    "Lace",
    "Tulle",
    "Velvet",
    "Jersey",
    "Crepe",
    "Organza",
    "Silk"
]

necklines = [
    "V Neck",
    "Sweetheart",
    "Square",
    "Boat",
    "High Neck",
    "Off Shoulder",
    "One Shoulder",
    "Illusion",
    "Strapless",
    "Halter"
]

sleeves = [
    "Sleeveless",
    "Cap Sleeve",
    "Short Sleeve",
    "Long Sleeve",
    "Puff Sleeve",
    "3/4 Sleeve"
]

lengths = [
    "Mini",
    "Knee",
    "Midi",
    "Tea",
    "Floor"
]

embellishments = [
    "Pleated",
    "Sequins",
    "Embroidery",
    "Beaded",
    "Feather",
    "Ruched",
    "Glitter",
    "Corset",
    "Draped",
    "None"
]

colors = [
    "White",
    "Ivory",
    "Black",
    "Red",
    "Pink",
    "Blue",
    "Royal Navy",
    "Sage",
    "Dusty Blue",
    "Emerald",
    "Gold",
    "Silver"
]

descriptions = []

for category in categories:
    for i in range(20):

        silhouette = random.choice(silhouettes)
        fabric = random.choice(fabrics)
        neckline = random.choice(necklines)
        sleeve = random.choice(sleeves)
        length = random.choice(lengths)
        embellishment = random.choice(embellishments)
        color = random.choice(colors)

        description = (
            f"{length.lower()} length {fabric.lower()} "
            f"{category.lower()} with {neckline.lower()} neckline, "
            f"{sleeve.lower()}, {embellishment.lower()} detailing "
            f"and {silhouette.lower()} silhouette in {color.lower()}."
        )

        descriptions.append({
            "description": description,
            "category": category,
            "silhouette": silhouette,
            "fabric": fabric,
            "neckline": neckline,
            "sleeve": sleeve,
            "length": length,
            "embellishment": embellishment,
            "color": color
        })

df = pd.DataFrame(descriptions)

os.makedirs("dataset", exist_ok=True)

df.to_csv("dataset/train.csv", index=False)

print(f"Dataset Created Successfully!")
print(f"Total Samples : {len(df)}")
print("Saved to dataset/train.csv")