# -*- encoding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
from itertools import permutations

import fire
import json
import random
import os

from llama import Llama, Dialog


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    n_few_shot_samples=1,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )


    os.environ['CUDA_VISIBLE_DEVICES'] = "0, 7"

    # Datasets paths
    examples_set_filename = "data/example_set.json"
    items_set_filename = "data/validation_set.json"

    # Load example set
    with open(examples_set_filename, "r") as f:
        examples = json.load(f)

    # Load news items to submit to the model
    with open(items_set_filename, "r") as f:
        data = json.load(f)

    shot_combinations = [(), (1, ), (2, 3)]

    for combination in shot_combinations:

        print("Combination: ", combination)
        shot_indices = list(combination)
        start_index = 0
        for i, el in enumerate(data[start_index:]):

            index = start_index + i

            print(f"element: {index}")

            first_shot = [
                    {
                        "role": "system",
                        "content": """
Quando ricevi in input il testo di una notizia di furto, devi restituire un JSON strutturato con i seguenti campi:

        • AUT, una lista di liste di stringhe. Ogni lista di stringhe deve contenere le parti del testo riportanti le informazioni relative a un singolo autore, quali il nome proprio o le iniziali e/o età, razza, etnia, residenza, abitante/nativo, sesso, occupazione, status giuridico (es. "incensurato", "pregiudicato", "gravato da precedenti"). Non vanno incluse altre caratteristiche o condizioni o ruoli (es. "malvivente", "ladro", "biondo", "marito", "moglie", "figlio", "ignoti".ecc)

        • AUTG, una lista di stringhe. Qui vanno inserite le parti di testo contenenti, se presenti, riferimenti all'intero gruppo di autori, nel caso gli autori del furto siano più di uno. Le informazioni da estrarre sono quelle socio-demografiche di riferimento quali età, razza, etnia, residenza, abitante/nativo, sesso, occupazione. Non includere termini generici ("autori", "ladri", "malviventi", "ignoti", ecc)

        • OBJ, una lista di liste di stringhe. Ogni lista deve contente la parte di testo che menziona un oggetto rubato e quella che, se presente, ne specifica la quantità. Non vanno inclusi oggetti o immobili danneggiati, né termini generici come "refurtiva", "bottino", "oggetti", "possedimenti", ecc.

        • VIC, una una lista di liste di stringhe. Ogni lista di stringhe deve contenere le parti del testo riportanti le informazioni relative a una singola vittima, quali il nome proprio o le iniziali e/o le informazioni socio-demografiche di riferimento, quali età, razza, etnia, residenza, abitante/nativo, sesso, occupazione. Non vanno incluse altre caratteristiche o condizioni o ruoli (es. "vittima", "proprietario", "biondo", "marito", "moglie", "figlio", ecc).

        • VICG, una lista di stringhe. Qui vanno inserite le parti di testo contenenti, se presenti, riferimenti all'intero gruppo di autori, nel caso le vittime del furto siano più di una. Le informazioni da estrarre sono quelle socio-demografiche di riferimento quali età, razza, etnia, residenza, abitante/nativo, sesso, occupazione. Non includere termini generici ("vittime", "proprietari", ecc).

        • PAR, lista di stringhe riferite a una parte lesa dal furto che non sia una persona fisica, ma un'attività commerciale (negozi, aziende, supermercati, ecc), un\'ente pubblico (comune, provincia, scuola, ecc) o un'associazione. La lista deve contenere la ragione sociale e la tipologia di questa entità. Vanno esclusi termini generici come "attività commerciale" o "impresa".

        • LOC, una lista di stringhe contentente le parti di testo riferite al luogo del delitto: città, zona cittadina (periferia o centro), via, numero civico, tipo di struttura ("abitazione", "casa", "appartamento", "condominio"). Nel caso il furto venga commesso a danno di un attività commerciale, va incluso anche il nome o tipo di tale attività che poi sarà presente anche nel campo "PAR".
"""
                    },
            ]

            next_shots_dialogs = []

            for shot_index in shot_indices:

                next_shots_dialogs.append({
                        "role": "user",
                        "content": shots[shot_index]['text']
                    })

                next_shots_dialogs.append({
                        "role": "assistant",
                        "content": json.dumps(shots[shot_index]['annotation'])
                    })

            final_request = {
                        "role": "user",
                        "content": el['text']
            }

            dialogs: List[Dialog] = [
                first_shot + next_shots_dialogs + [final_request]
            ]

           # for i, el in enumerate(dialogs[0]):
           #     print(f"Len shot {i}:", len(el['content']))

            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            for dialog, result in zip(dialogs, results):
                el['completion'] = result['generation']['content']

            char_str = [str(index) for index in shot_indices]
            with open(f"predicted_{len(combination)}_{''.join(char_str)}.json", "w") as f:
                json.dump(data, f)


if __name__ == "__main__":
    fire.Fire(main)
