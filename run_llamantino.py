from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

import torch
import json
import os

# Set GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = "6, 7"

# Load Model
model_name = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
# model_name = "meta-llama/Llama-2-13b-chat-hf"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

quantization = False

if quantization:
    # quantization configuration
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
else:
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")


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
examples_combinations = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 8), (0, 9), (1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 8), (1, 9), (2, 0), (2, 1), (2, 3), (2, 4),
         (2, 5), (2, 8), (2, 9), (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 8), (3, 9), (4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (4, 8), (4, 9), (5, 0),
         (5, 1), (5, 2), (5, 3), (5, 4), (5, 8), (5, 9), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 9), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4),
         (9, 5), (9, 8), (3, 4, 2), (2, 3, 4), (4, 2, 3), (3, 1, 9), (1, 3, 9), (3, 9, 1), (4, 9, 8), (9, 8, 4), (4, 8, 9), (3, 5, 0), (0, 3, 5), (5, 0, 3), (9, 0, 2),
         (2, 9, 0), (0, 2, 9), (3, 9, 2), (2, 9, 3), (9, 2, 3), (1, 4, 9), (4, 1, 9), (4, 9, 1), (2, 1, 3), (1, 3, 2), (2, 3, 1), (9, 2, 4), (2, 4, 9),
         (2, 9, 4), (8, 2, 0), (2, 8, 0), (2, 0, 8), (5, 8, 9), (9, 8, 5), (5, 9, 8), (1, 2, 4), (2, 1, 4), (4, 1, 2), (0, 8, 4), (8, 4, 0), (8, 0, 4),
         (9, 8, 0), (8, 0, 9), (8, 9, 0), (2, 1, 0), (2, 0, 1), (1, 0, 2), (1, 3, 5), (5, 1, 3), (1, 5, 3), (2, 0, 5), (5, 2, 0), (2, 5, 0), (3, 9, 5),
         (5, 3, 9), (9, 5, 3), (8, 5, 2), (2, 8, 5), (8, 2, 5), (1, 0, 4), (4, 1, 0), (0, 1, 4), (4, 3, 9), (4, 9, 3), (3, 4, 9), (0, 4, 5), (0, 5, 4),
         (5, 0, 4), (2, 4, 0), (4, 0, 2), (0, 4, 2),(8, 9, 2, 1), (1, 9, 2, 8), (2, 9, 1, 8), (2, 3, 9, 0), (2, 9, 3, 0), (2, 0, 9, 3), (4, 1, 0, 9), (0, 1, 9, 4), (9, 4, 0, 1), (1, 8, 5, 2),
         (8, 1, 2, 5), (2, 8, 1, 5), (9, 8, 4, 0), (0, 8, 9, 4), (4, 8, 0, 9), (4, 5, 3, 9), (5, 4, 3, 9), (9, 4, 3, 5), (4, 9, 2, 3), (3, 2, 4, 9),
         (2, 9, 3, 4), (8, 9, 0, 1), (1, 0, 9, 8), (0, 9, 1, 8), (0, 2, 9, 4), (9, 0, 4, 2), (2, 4, 0, 9), (0, 2, 4, 3), (0, 4, 3, 2), (2, 3, 0, 4),
         (0, 3, 9, 1), (3, 1, 0, 9), (1, 3, 0, 9), (0, 5, 9, 2), (5, 0, 2, 9), (9, 0, 5, 2), (3, 0, 2, 5), (0, 2, 5, 3), (3, 0, 5, 2), (4, 1, 5, 3),
         (5, 4, 3, 1), (3, 4, 1, 5), (3, 4, 8, 9), (3, 9, 4, 8), (4, 3, 9, 8), (1, 3, 4, 2), (1, 4, 3, 2), (4, 2, 1, 3), (5, 1, 0, 2), (0, 1, 2, 5),
         (5, 2, 1, 0), (0, 4, 1, 5), (4, 5, 1, 0), (4, 0, 5, 1), (2, 5, 8, 3), (2, 8, 5, 3), (3, 8, 5, 2), (4, 1, 9, 8), (1, 9, 4, 8), (4, 1, 8, 9),
         (5, 2, 4, 8), (4, 8, 2, 5), (8, 5, 2, 4), (3, 4, 8, 5), (3, 5, 8, 4), (4, 3, 8, 5), (9, 1, 0, 5), (5, 9, 0, 1), (5, 1, 0, 9)]

for combination in examples_combinations:
    print("combination: ", combination)
    examples_indices = list(combination)

    # index of the starting item
    start_index = 0
    for i, el in enumerate(data[start_index:]):

        index = start_index + i

        print(f"element: {index}")

        first_text = """
Quando ricevi in input il testo di una notizia di furto, devi restituire un JSON strutturato con i seguenti campi:

    • AUT, una lista di liste di stringhe. Ogni lista di stringhe deve contenere le parti del testo riportanti le informazioni relative a un singolo autore, quali il nome proprio o le iniziali e/o età, razza, etnia, residenza, abitante/nativo, sesso, occupazione, status giuridico (es. "incensurato", "pregiudicato", "gravato da precedenti"). Non vanno incluse altre caratteristiche o condizioni o ruoli (es. "malvivente", "ladro", "biondo", "marito", "moglie", "figlio", "ignoti".ecc)

    • AUTG, una lista di stringhe. Qui vanno inserite le parti di testo contenenti, se presenti, riferimenti all'intero gruppo di autori, nel caso gli autori del furto siano più di uno. Le informazioni da estrarre sono quelle socio-demografiche di riferimento quali età, razza, etnia, residenza, abitante/nativo, sesso, occupazione. Non includere termini generici ("autori", "ladri", "malviventi", "ignoti", ecc)

    • OBJ, una lista di liste di stringhe. Ogni lista deve contente la parte di testo che menziona un oggetto rubato e quella che, se presente, ne specifica la quantità. Non vanno inclusi oggetti o immobili danneggiati, né termini generici come "refurtiva", "bottino", "oggetti", "possedimenti", ecc.

    • VIC, una una lista di liste di stringhe. Ogni lista di stringhe deve contenere le parti del testo riportanti le informazioni relative a una singola vittima, quali il nome proprio o le iniziali e/o le informazioni socio-demografiche di riferimento, quali età, razza, etnia, residenza, abitante/nativo, sesso, occupazione. Non vanno incluse altre caratteristiche o condizioni o ruoli (es. "vittima", "proprietario", "biondo", "marito", "moglie", "figlio", ecc).

    • VICG, una lista di stringhe. Qui vanno inserite le parti di testo contenenti, se presenti, riferimenti all'intero gruppo di autori, nel caso le vittime del furto siano più di una. Le informazioni da estrarre sono quelle socio-demografiche di riferimento quali età, razza, etnia, residenza, abitante/nativo, sesso, occupazione. Non includere termini generici ("vittime", "proprietari", ecc).

    • PAR, lista di stringhe riferite a una parte lesa dal furto che non sia una persona fisica, ma un'attività commerciale (negozi, aziende, supermercati, ecc), un\'ente pubblico (comune, provincia, scuola, ecc) o un'associazione. La lista deve contenere la ragione sociale e la tipologia di questa entità. Vanno esclusi termini generici come "attività commerciale" o "impresa".

    • LOC, una lista di stringhe contentente le parti di testo riferite al luogo del delitto: città, zona cittadina (periferia o centro), via, numero civico, tipo di struttura ("abitazione", "casa", "appartamento", "condominio"). Nel caso il furto venga commesso a danno di un attività commerciale, va incluso anche il nome o tipo di tale attività che poi sarà presente anche nel campo "PAR".
"""
        messages = [{
                        "role": "system",
                        "content": first_text
        }]

        for example_index in examples_indices:

            messages.append({
                    "role": "user",
                    "content": examples[example_index]['text']
                })

            messages.append({
                    "role": "assistant",
                    "content": json.dumps(examples[example_index]['annotation'])
                })

        messages.append({
                    "role": "user",
                    "content": el['text']
        })

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

        outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=True, top_p=0.9, temperature=0.6)
        results = tokenizer.batch_decode(outputs)[0]
        data[index]['completion'] = results

        # Output_file: predicted_<n° shots>_<examples combination>.json
        char_str = [str(index) for index in examples_indices ]
        with open(f"predictions/predicted_{len(combination)}_{''.join(char_str)}.json", "w") as f:
            json.dump(data, f)
