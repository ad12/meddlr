from meddlr.utils.env import get_available_gpus

if len(get_available_gpus()) > 0:
    print(get_available_gpus()[0])
