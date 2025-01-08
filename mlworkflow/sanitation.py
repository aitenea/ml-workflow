from pandas import DataFrame


def check_str(obj):
    if not isinstance(obj, str):
        raise TypeError(f"string expected, got {type(obj).__name__}")


def check_strs(*args):
    for a in args:
        check_str(a)


def check_pd(obj):
    if not isinstance(obj, DataFrame):
        raise TypeError(f"pandas dataframe expected, got {type(obj).__name__}")


def check_var(df, feat):
    if feat not in df:
        raise KeyError(f"variable {feat} missing inside dataframe")


def check_vars(df, feats):
    for f in feats:
        check_var(df, f)


def is_float(obj):
    res = True
    try:
        float(obj)
    except ValueError:
        res = False

    return res


def is_real_valued(obj):
    res = True
    if not all(list(map(is_float, obj))):
        res = False

    return res
