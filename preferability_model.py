import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate City Preferability.")
    parser.add_argument("--data", type=str, required=True, help="path to the data file")
    args = parser.parse_args()

    city = args.city
    criteria = args.criteria

    # Placeholder for the evaluation logic

if __name__ == "__main__":
    main()
