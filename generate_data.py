import numpy as np
import pandas as pd
import random

def generate_housing_data(rows=1000, filename="housing.csv"):
    np.random.seed(42)

    locations = {
        "City": {"base_price": 6000, "lat": 19.07, "lon": 72.87},
        "Suburb": {"base_price": 4500, "lat": 19.20, "lon": 72.95},
        "Rural": {"base_price": 2500, "lat": 19.50, "lon": 73.10}
    }

    data = []

    for _ in range(rows):
        location = random.choice(list(locations.keys()))
        base = locations[location]["base_price"]

        area = np.random.randint(500, 3000)
        bedrooms = np.random.randint(1, 6)
        bathrooms = np.random.randint(1, 5)
        parking = np.random.randint(0, 3)
        age = np.random.randint(0, 30)
        furnished = np.random.choice([0, 1])

        # Realistic pricing formula
        price = (
            area * base
            + bedrooms * 200000
            + bathrooms * 150000
            + parking * 100000
            - age * 50000
            + furnished * 300000
        )

        # Add market noise
        noise = np.random.normal(0, 500000)
        price += noise

        latitude = locations[location]["lat"] + np.random.normal(0, 0.01)
        longitude = locations[location]["lon"] + np.random.normal(0, 0.01)

        data.append([
            area,
            bedrooms,
            bathrooms,
            parking,
            age,
            furnished,
            location,
            round(latitude, 6),
            round(longitude, 6),
            int(price)
        ])

    columns = [
        "area",
        "bedrooms",
        "bathrooms",
        "parking",
        "age",
        "furnished",
        "location",
        "latitude",
        "longitude",
        "price"
    ]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)

    print(f"Dataset generated with {rows} rows → {filename}")


if __name__ == "__main__":
    generate_housing_data(rows=1000)  # Change to any number