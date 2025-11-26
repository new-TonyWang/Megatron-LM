from transformer_engine.pytorch.module import  load_svd_history, no_use_metis
from contextlib import ExitStack
def metis_gradacc_broadcast_func():
    stack = ExitStack()
    stack.enter_context(load_svd_history())
    return stack

def no_use_and_low_rank_metis_forward_func():
    stack = ExitStack()
    stack.enter_context(no_use_metis())
    return stack