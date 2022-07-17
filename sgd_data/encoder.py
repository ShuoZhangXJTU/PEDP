import numpy as np


def filter_da(da):
    return da


def state_vectorize(state, config):
    """
    state: dict_keys(['user_action', 'sys_action', 'user_goal', 'belief_state', 'history', 'others'])
    state_vec: [user_act, last_sys_act, inform, request, book, degree]
    """
    user_act = np.zeros(len(config.da_usr))
    for da in state['user_action']:
        da = filter_da(da)
        user_act[config.dau2idx[da]] = 1.

    last_sys_act = np.zeros(len(config.da))
    if 'last_sys_action' in state.keys():
        for da in state['last_sys_action']:
            da = filter_da(da)
            last_sys_act[config.da2idx[da]] = 1.

    user_history = np.zeros(len(config.da_usr))
    for da in state['history']['user']:
        da = filter_da(da)
        user_history[config.dau2idx[da]] = 1.

    sys_history = np.zeros(len(config.da))
    for da in state['history']['sys']:
        da = filter_da(da)
        sys_history[config.da2idx[da]] = 1.

    inform = np.zeros(len(config.inform_da))
    for domain in state['belief_state']['INFORM']:
        for slot, value in state['belief_state']['INFORM'][domain].items():
            dom_slot = domain + '-' + slot
            inform[config.inform2idx[dom_slot]] = 1.

    request = np.zeros(len(config.request_da))
    for domain in state['belief_state']['REQUEST']:
        for slot in state['belief_state']['REQUEST'][domain]:
            dom_slot = domain + '-' + slot
            request[config.request2idx[dom_slot]] = 1.

    final = 1. if state['others']['terminal'] else 0.
    state_vec = np.r_[user_act, last_sys_act, user_history, sys_history, inform, request, final]
    assert len(state_vec) == config.s_dim
    return state_vec


def action_vectorize(action, config, provided_act=None):
    act_vec = np.zeros(config.a_dim)
    used_act = provided_act if provided_act is not None else action['sys_action']
    for da in used_act:
        da = filter_da(da)
        act_vec[config.da2idx[da]] = 1
    return act_vec


def action_seq(action, config, idx_only=False, provided_act=None):
    act_seq = []
    used_act = provided_act if provided_act is not None else action['sys_action']
    for da in used_act:
        da = filter_da(da)
        act_seq.append(config.da2idx[da])
    if idx_only:
        return act_seq
    act_seq.insert(0, 1)  # SOS
    act_seq.append(2)  # EOS
    act_seq = pad_to(config.max_len, act_seq, True)
    return act_seq


def pad_to(max_len, tokens, do_pad=True, pad_id=0):
    if len(tokens) >= max_len:
        return tokens[0:max_len - 1] + [tokens[-1]]
    elif do_pad:
        return tokens + [pad_id] * (max_len - len(tokens))
    else:
        return tokens
