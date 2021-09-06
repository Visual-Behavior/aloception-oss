def DLtoLD(d):
    """
    Transform dict of lists to a list of dicts
    """
    if not d:
        return []
    # reserve as much *distinct* dicts as the longest sequence
    result = [{} for i in range(max(map(len, d.values())))]
    # fill each dict, one key at a time
    for k, seq in d.items():
        for oneDict, oneValue in zip(result, seq):
            oneDict[k] = oneValue
    return result


def LDtoDL(LD):
    """Transform a list of dict to a dict of list"""
    DL = {}
    for d in LD:
        for k, v in d.items():
            if k not in DL:
                DL[k] = [v]
            else:
                DL[k].append(v)
    return DL
