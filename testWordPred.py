import word_pred

while True:
    input_seq = input("Enter word: ")
    if input_seq == 'q':
        break

    suggested_words = word_pred.predict(input_seq)

    if suggested_words:
        print("Suggested words:")
        print(suggested_words)
    else:
        print("No suggestions found.")