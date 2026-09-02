import pandas as pd

df = pd.read_csv("../data-bases/pokemon.csv", index_col="Name")

while True:
    print("Enter a Pokemon name for details")
    print("OR")
    print("Enter Exit to quit Search")

    searchName = input("Entry: ")

    if searchName.lower() == "exit":
        break

    try:
        print(df.loc[searchName])
    except KeyError:
        print(f"{searchName} not found")