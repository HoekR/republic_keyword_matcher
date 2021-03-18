import pandas as pd
from .datamangler import hypothetical_life, make_abbreviated_delegates
from .abbr_delegates_to_pandas import abbreviated_delegates_from_excel
import pkg_resources

#DATA_PATH = pkg_resources.resource_filename('__name__', 'data/csvs')
PICKLE_FILE = pkg_resources.resource_filename(__name__, 'csvs/abbreviated_delegates.pickle')

def get_db():
    try:
        abbreviated_delegates = pd.read_pickle(PICKLE_FILE)
    except IOError:
        try:
            abbreviated_delegates = abbreviated_delegates_from_excel()
        except IOError:
            abbreviated_delegates = make_abbreviated_delegates()
    return abbreviated_delegates

# if not pd.isna(geboortejaar) and geboortejaar.year < day and sterfjaar.year > day:
#     age_at_repr = len(pd.period_range(geboortejaar, day, freq="Y"))  # cool, though this gives impossible ages :-)
# else:
#     age_at_repr = pd.nan
