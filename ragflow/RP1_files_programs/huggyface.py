#Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, set_seed

#set seed
set_seed(33)

#choose model from transformer package
generator = pipeline("text-generation", model="facebook/opt-2.7b", do_sample=True)


# Generate text
print(generator("Is Timo a good person? He once tried to fight a guy who spilled a beer on him."))

#run with python3.9