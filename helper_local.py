def import_wandb():
    try:
        import wandb
        from private_login import wandb_login
        wandb_login()
        return wandb
    except ImportError:
        return None

