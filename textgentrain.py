from textgenrnn import textgenrnn

textgen = textgenrnn(name="model2")

textgen.reset()
textgen.train_from_largetext_file('dadjokes.txt', new_model=True, num_epochs=500,
                                  word_level=True,
                                  max_length=10,
                                  max_gen_length=10,
                                  max_words=50)
textgen.generate(1)
print(textgen.model.summary())
