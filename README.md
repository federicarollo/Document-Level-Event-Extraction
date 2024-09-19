# Document Level Event Extraction from Italian Crime News Using Minimal Data

This repository contains the reference code for the paper _[Document Level Event Extraction from Italian Crime News Using Minimal Data](https://)_.

<!-- Please cite with the following BibTeX:
```
@inproceedings{}
``` -->

## Overview
**Document-level Event Extraction** (DEE) aims at identifying information about an event within a lengthy text. This usually refers to any significant occurrence, action, or situation involving specific entities (individuals, organizations, event-specific roles, abstractions like laws or policies...) and unfolding over a specific period of time.

Event extraction in daily crime news involves sifting through unstructured data to identify and categorize events accurately. Also, it aids in detection of emerging crime trends and threats, enabling risk reduction and community safety.

<!-- Space for the figure -->

However, developing these systems faces challenges, especially in acquiring labeled datasets, as the annotation process requires precise guidelines and specialized expertise. Leveraging minimally informed methods for crime news analysis can alleviate these challenges, reducing dependence on annotated datasets while maintaining accuracy in event detection, categorization, and linking.

We establish a methodology to enable LLMs to accurately extract event related data and provide it in a structured JSON format.

## Environment setup
Please use Python>=3.8 as well as the following packages:
```
torch
transformers
accelerate
bitsandbytes
mistral_common
```

## Extraction

Run the script "run_mistral_models.py" to execute extraction with Mistral models.
```
python run_mistral_models.py
```

Run the script "run_llama2_models.py" to execute extraction with Llama 2 models.
```
python run_llama2_models.py
```

## Evaluation
To get evaluation results, run the notebook "evaluation.ipynb"
