# from .BackdoorWM import BackdoorWM
# from .ImperceptibleWM import ImperceptibleWM
# from .ImperceptibleWM2 import ImperceptibleWM2
# from .RandomWM import RandomWM
# from .SurviveWM import SurviveWM
# from .SurviveWM2 import SurviveWM2
# from .atom.ATOM import ATOM
# from .Integrity import QueryBasedVerificationDefense as IntegrityVerification
# from .Revisiting import Revisiting

# __all__ = [
#     'BackdoorWM',
#     'ImperceptibleWM',
#     'ImperceptibleWM2',
#     'RandomWM',
#     'SurviveWM',
#     'SurviveWM2',
#     'ATOM',
#     'IntegrityVerification',
#     'Revisiting'
# ]
# Defense module for link prediction
# Original node classification imports are commented out to avoid dependency issues

# Link prediction defenses
try:
    from .linkpred_defenses import (
        RandomWMLinkPred,
        BackdoorWMLinkPred,
        SurviveWMLinkPred,
        ImperceptibleWMLinkPred,
        IntegrityLinkPred
    )
    
    __all__ = [
        'RandomWMLinkPred',
        'BackdoorWMLinkPred',
        'SurviveWMLinkPred',
        'ImperceptibleWMLinkPred',
        'IntegrityLinkPred'
    ]
except ImportError as e:
    print(f"Warning: Could not import link prediction defenses: {e}")
    __all__ = []