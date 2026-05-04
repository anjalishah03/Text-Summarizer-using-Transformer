# Dialogue Summarization with Fine-tuned T5

This repository contains code for fine-tuning a T5-small model for dialogue summarization using the SAMSum dataset. The model is trained to generate concise summaries from conversational text.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Usage](#usage)
- [Results](#results)

## Project Overview
This project focuses on leveraging the Transformer architecture, specifically the T5-small model, for the task of dialogue summarization. Dialogue summarization is crucial for distilling long conversations into short, informative summaries, which can be beneficial in various applications like customer support, meeting minutes, and personal communication analysis.

## Setup Instructions
To set up the environment and run the notebook, follow these steps:

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install necessary libraries:**
    ```bash
    !pip install transformers
    !pip install "transformers[torch]"
    !pip install pandas
    ```

3.  **Download the dataset:**
    Ensure `samsum-train.csv` and `samsum-validation.csv` are available in your working directory. (These are typically provided or linked in the original source of the notebook).

## Dataset
The project uses the [SAMSum Dataset](https://huggingface.co/datasets/samsum), which consists of thousands of dialogue transcripts accompanied by human-written summaries.

## Model Training
The T5-small model is fine-tuned using the Hugging Face Transformers library. Key steps in the training process include:
-   Data cleaning and preprocessing.
-   Tokenization of dialogues and summaries using `T5Tokenizer`.
-   Configuration of `TrainingArguments` (e.g., batch size, epochs).
-   Training the model using `Trainer`.

## Usage
After training (or loading the pre-trained model), you can use the `summarize_dialogue` function to generate summaries for new dialogues.

### Example:
```python
test_dialogue = """Ollie: Hi , are you in Warsaw Jane: yes, just back! Btw are you free for diner the 19th? Ollie: nope! Jane: and the 18th? Ollie: nope, we have this party and you must be there, remember? Jane: oh right! i lost my calendar.. thanks for reminding me Ollie: we have lunch this week? Jane: with pleasure! Ollie: friday? Jane: ok Jane: what do you mean " we don't have any more whisky!" lol.. Ollie: what!!! Jane: you just call me and the all thing i heard was that sentence about whisky... what's wrong with you? Ollie: oh oh... very strange! i have to be carefull may be there is some spy in my mobile! lol Jane: dont' worry, we'll check on friday. Ollie: don't forget to bring some sun with you Jane: I can't wait to be in Morocco.. Ollie: enjoy and see you friday Jane: sorry Ollie, i'm very busy, i won't have time for lunch tomorrow, but may be at 6pm after my courses?this trip to Morocco was so nice, but time consuming! Ollie: ok for tea! Jane: I'm on my way.. Ollie: tea is ready, did you bring the pastries? Jane: I already ate them all... see you in a minute Ollie: ok"""
summary = summarize_dialogue(test_dialogue)
print("Summary:", summary)
```

## Results
The fine-tuned model demonstrates the ability to generate coherent and concise summaries of dialogues. Further evaluation metrics (like ROUGE scores) can be incorporated for a more comprehensive assessment of the model's performance.
