import sys
import pickle

def print_first_10_lines(pickle_file):
    # Open the pickle file in binary mode
    with open(pickle_file, 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)
        print(type(data))
    

        # Print the first 10 key-value pairs
        count = 0
        for key, value in data.items():
            if count >= 100:
                break
            print(f"Key: {key}, Value: {value}")
            count += 1

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python program.py <pickle_file>")
        sys.exit(1)

    # Get the name of the pickle file from command-line arguments
    pickle_file = sys.argv[1]

    # Call the function to print the first 10 lines
    print_first_10_lines(pickle_file)