from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

import torch
import json
import os

# Set GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

# Load Model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Load tokenizer
tokenizer = MistralTokenizer.v1()

quantization = True

if quantization:
    # quantization configuration
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
else:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.to("cuda")

# Datasets paths
examples_set_filename = "data/example_set.json"
items_set_filename = "data/validation_set.json"

# Load example set
with open(examples_set_filename, "r") as f:
    examples = json.load(f)

# Load news items to submit to the model
with open(items_set_filename, "r") as f:
    data = json.load(f)

# Insert example combinations. For zero shot, insert ()
examples_combinations = [(0, 4, 5, 3)]

for combination in examples_combinations:
    print("combination: ", combination)
    example_indices = list(combination)

    # index of the starting item
    start_index = 0
    for i, el in enumerate(data[start_index:]):

        index = start_index + i

        print(f"element: {index}")

        news_to_insert = {}
        if combination == ():
            # If zero shot, insert the text of the news to extract
            news_to_insert = el
        else:
            # If not zero shot, insert the text of the first example
            news_to_insert = examples[example_indices[0]]

        first_text = f"""
Ecco il testo di una notizia di furto:

    {news_to_insert['text']}

Devi restituire esclusivamente un JSON strutturato con i seguenti campi:
    • AUT, una lista di liste di stringhe. Ogni lista di stringhe deve contenere le parti del testo riportanti le informazioni relative a un singolo autore, quali il nome proprio o le iniziali e/o età, razza, etnia, residenza, abitante/nativo, sesso, occupazione, status giuridico (es. "incensurato", "pregiudicato", "gravato da precedenti"). Non vanno incluse altre caratteristiche o condizioni o ruoli (es. "malvivente", "ladro", "biondo", "marito", "moglie", "figlio", "ignoti" ecc)

    • AUTG, una lista di stringhe. Qui vanno inserite le parti di testo contenenti, se presenti, riferimenti all'intero gruppo di autori, nel caso gli autori del furto siano più di uno. Le informazioni da estrarre sono quelle socio-demografiche di riferimento quali età, razza, etnia, residenza, abitante/nativo, sesso, occupazione. Non includere termini generici ("autori", "ladri", "malviventi", "ignoti", ecc)

    • OBJ, una lista di liste di stringhe. Ogni lista deve contente la parte di testo che menziona un oggetto rubato e quella che, se presente, ne specifica la quantità. Non vanno inclusi oggetti o immobili danneggiati, né termini generici come "refurtiva", "bottino", "oggetti", "possedimenti", ecc

    • VIC, una una lista di liste di stringhe. Ogni lista di stringhe deve contenere le parti del testo riportanti le informazioni relative a una singola vittima, quali il nome proprio o le iniziali e/o le informazioni socio-demografiche di riferimento, quali età, razza, etnia, residenza, abitante/nativo, sesso, occupazione. Non vanno incluse altre caratteristiche o condizioni o ruoli (es. "vittima", "proprietario", "biondo", "marito", "moglie", "figlio", ecc).

    • VICG, una lista di stringhe. Qui vanno inserite le parti di testo contenenti, se presenti, riferimenti all'intero gruppo di autori, nel caso le vittime del furto siano più di una. Le informazioni da estrarre sono quelle socio-demografiche di riferimento quali età, razza, etnia, residenza, abitante/nativo, sesso, occupazione. Non includere termini generici ("vittime", "proprietari", ecc).

    • PAR, lista di stringhe riferite a una parte lesa dal furto che non sia una persona fisica, ma un'attività commerciale (negozi, aziende, supermercati, ecc), un'ente pubblico (comune, provincia, scuola, ecc) o un'associazione. La lista deve contenere la ragione sociale e la tipologia di questa entità. Vanno esclusi termini generici come "attività commerciale" o "impresa".

    • LOC, una lista di stringhe contentente le parti di testo riferite al luogo del delitto: città, zona cittadina (periferia o centro), via, numero civico, tipo di struttura ("abitazione", "casa", "appartamento", "condominio"). Nel caso il furto venga commesso a danno di un attività commerciale, va incluso anche il nome o tipo di tale attività che poi sarà presente anche nel campo "PAR".
"""

        messages = [
                        UserMessage(content=first_text),
                        AssistantMessage(content=json.dumps(news_to_insert['annotation']))
                    ]

        if combination != ():
            for example_index in example_indices[1:]:

                messages.append(UserMessage(content=f"""
Bene! Ora fai lo stesso per il testo la seguente notizia:
    {examples[example_index]['text']}
    """))

                messages.append(AssistantMessage(content=json.dumps(examples[example_index]['annotation'])))

            messages.append(UserMessage(content=f"""
Bene! Ora fai lo stesso per il testo la seguente notizia:
    {el['text']}
    """))

        completion_request = ChatCompletionRequest(messages=messages)

        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        encoded = torch.tensor([tokens], dtype=torch.int32).to('cuda')

        generated_ids = model.generate(encoded, max_new_tokens=1000, do_sample=True)
        data[index]['completion'] = tokenizer.decode(generated_ids[0].tolist())

        # Output_file: predicted_<n° shots>_<examples combination>.json
        char_str = [str(index) for index in example_indices]
        with open(f"predicted_{len(combination)}_{''.join(char_str)}.json", "w") as f:
            json.dump(data, f)
