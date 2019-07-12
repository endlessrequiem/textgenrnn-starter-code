from textgenrnn import textgenrnn

textgen = textgenrnn(weights_path='model1_weights.hdf5',
                     vocab_path='model1_vocab.json',
                     config_path='model1_config.json',)

textgen.generate(1, temperature=[1.0, 1.2, 1.5])
