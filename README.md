# Context injection in LLM-based lip reading
**Coursework project in Ecole polytechnique**

## Posters
[Posters](posters/v2.pdf)

## Report 
**On the way...**

## Acknowledgements
This project is largely based on [VSP-LLM](https://github.com/Sally-SH/VSP-LLM?tab=readme-ov-file).

## Repository descriptions
The datasets require a **lot** of preprocessing.   
In particular, we did the following preprocessing:  
1. Download the LRS2 dataset (you need to wait for consent from BBC) or the MultiVSR dataset (you need to scrape from youtube).  
(Since the LRS3 dataset is no longer available, we have to make do with these two.)  
If you are using MultiVSR, you can refer to the `vsp_llm_preprocess_extra` files to see how we filter clips.  
**Due to copyright reasons we can't directly provide the processed dataset.**  
2. Follow the instructions in [VSP-LLM](https://github.com/Sally-SH/VSP-LLM?tab=readme-ov-file) about preprocessing (4 steps to make dataset and 4 steps to make VSP-LLM-style token deduplication counts).
In particular, you need to use the `avhubert_preprocess_override` files provided in our repository to override those from the AV-HuBERT repository.
3. Use `train.sh` and `decode.sh` from `VSL_LLM/scripts` to train and infer on LRS2; use `train_multivsr.sh` and `decode_multivsr.sh` to train and infer on MultiVSR.