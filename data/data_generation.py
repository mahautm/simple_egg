import random
from sklearn.model_selection import train_test_split


def gen_data(
    max_symbol=100,
    training_path="training_100.txt",
    validation_path="validation_100.txt",
):
    numbers = []
    for i in range(max_symbol):
        for j in range(max_symbol):
            numbers.append(str(i) + " " + str(j))
    train, test = train_test_split(numbers, test_size=0.3)

    text_train = ""
    for t in train:
        text_train += t + "\n"

    text_test = ""
    for t in test:
        text_test += t + "\n"

    with open(training_path, "w") as f:
        f.write(text_train)
    with open(validation_path, "w") as f:
        f.write(text_test)


if __name__ == "__main__":
    gen_data()
