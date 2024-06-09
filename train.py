from happytransformer import HappyGeneration, GENTrainArgs

happy_gen = HappyGeneration("LLAMA", "mistralai/Mistral-7B-v0.1")
args = GENTrainArgs(num_train_epochs=1)
happy_gen.train("train.txt", args=args)
args = GENSettings(max_length=2000)
result = happy_gen.generate_text("<|system|>Generate a 2000-character lesson on getting help for suicidal ideation</s><|assistant|>", args=args)
happy_gen.save("model/")