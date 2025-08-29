try:
    from IPython import get_ipython

    ipy = get_ipython()
    if ipy is not None:
        shell = ipy.__class__.__name__
        if shell == "ZMQInteractiveShell":
            IN_JUPYTER = True
        else:
            IN_JUPYTER = False
    else:
        IN_JUPYTER = False
except (NameError, ImportError, AttributeError):
    IN_JUPYTER = False
