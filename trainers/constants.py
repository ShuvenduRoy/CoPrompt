def get_dataset_specified_config(dataset):
    """Get dataset specific."""
    cfg = {
        "ImageNet": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 8.0,
        },
        "Caltech101": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 4.0,
        },
        "OxfordPets": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 2,
            "TRAINER.W": 0.1,
        },
        "StanfordCars": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 4.0,
        },
        "OxfordFlowers": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 9,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 8.0,
        },
        "Food101": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 9,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 0.5,
            "OPTIM.MAX_EPOCH": 5,
        },
        "FGVCAircraft": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 9,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 2.0,
        },
        "SUN397": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 8.0,
        },
        "DescribableTextures": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 2,
            "TRAINER.W": 8.0,
        },
        "EuroSAT": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 2,
            "TRAINER.W": 0.1,
        },
        "UCF101": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 2.0,
        },
    }.get(dataset, {})

    return " ".join([f"{k} {v}" for k, v in cfg.items()]).split(" ")
