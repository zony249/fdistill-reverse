# Title 

### Important Environment Details 
- Python: `3.9.19` 
- Huggingface: `4.1.0`
- PyTorch: `2.3.0`
- PyTorch-Lightning: `1.0.4`

## Classification 

1. Download dataset or model if necessary (i.e. if compute node is not connected to the internet): 
    ```bash 
    python3 download-dataset-for-cc.py 
    python3 model-download-for-cc.py
    ```
    Before any training, you should also consider making a `runs` and `models` directory to contain experiments and hold onto models. 

2. Finetune teacher: 
    ``` bash 
    bash scripts/run_finetune_teacher.sh
    ```
3. Distil student models from teacher:
    - Run any experiment in any following folder under `scripts/`: 
        - `all-to-one`
        - `depth`
        - `forward`
        - `reverse`
        - `mle`
        - `random-shuffle`
        - `width`
    - Be sure to run them from the `classification` directory. For example: 
        ```bash 
        bash scripts/all-to-one/mnli-all-one-random.sh
        ```
4. Evaluate the model. Make sure you use the best model saved under `best_tfmr` of each experiment.

    ```bash 

    bash scripts/eval.sh
    ```
5. Distance and Angles Analysis: 
    ```bash 
    bash scripts/run_dist_analysis.sh
    ```

## Data to Text / Machine Translation 

Make sure you are in the respective directories: `dart` for Data to Text or `t5mt` for Machine Translation.

1. Download dataset:
    - DART: [a google drive link that will be revealed once anonymity is lifted](a_link)
    - WMT16 En--Ro: [a google drive link that will be revealed once anonymity is lifted](a_link)

    Before any training, you should also consider making a `runs` and `models` directory to contain experiments and hold onto models.

2. Finetune teacher: 

    ```bash

    bash scripts/run_finetune_teacher.sh 
    ```
3. Distil student models from teacher:
    - Run any experiment in any following folder under `scripts/`: 
        - `all-to-one`
        - `width`
        - `forward`
        - `reverse`
        - `mle`
        - `random-shuffle`
    - Be sure to run them from the `classification` directory. For example: 
        ```bash 
        bash scripts/all-to-one/all-one-3l-random.sh 
        ```
4. Evaluate the model. Make sure you use the best model saved under `best_tfmr` of each experiment.
    - DART: 
        1.  first run `run_eval.sh`, which generates predictions stored in `res.out`. 
        2.  Then feed `res.out` into `run_eval_on_dart.sh`
    - WMT16 En--Ro: 
        1.  run `run_eval.sh`
5. Distance and Angles Analysis: 
    - WMT16 En--Ro: 
        ```bash 
        bash scripts/run_dist_analysis.sh
        ```  