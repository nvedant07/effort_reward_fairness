import shelve

def load_params(store_loc, key, param_ranges):
    with shelve.open(store_loc) as param_db:
        old_param_ranges = param_db.get(get_key(key, 'ranges'))
        best_params = param_db.get(get_key(key, 'params'))

        assert (old_param_ranges is None and best_params is None) \
                or (old_param_ranges is not None and best_params \
                is not None)

        if best_params is None:
            return None
        # check if the old and the current param ranges are equal
        # if not then the optimal parameters need to be recomputed
        if old_param_ranges != param_ranges:
            return None
        return best_params

def save_params(store_loc, key, param_ranges, best_params):
    with shelve.open(store_loc) as param_db:
        param_db[get_key(key, 'ranges')] = param_ranges
        param_db[get_key(key, 'params')] = best_params

def get_key(key, key_type):
    if key_type == 'ranges':
        suffix = '_ranges'
    elif key_type == 'params':
        suffix = '_best_params'
    else:
        raise ValueError('unsupported key type "{}"'.format(key_type))
    return key + suffix

