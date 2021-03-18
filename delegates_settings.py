from republic.fuzzy.fuzzy_keyword_searcher import FuzzyKeywordSearcher
from republic.config import republic_config


base_config = republic_config.base_config
naaminput = ['Van Maasdam', 'Cammingha', 'Haersolte', 'Iddekinge', 'Isselmuden', 'van Welderen',
             'Ockersse', 'Heukelum', 'Rose', 'Rouse', 'van Singendonck', 'Zanders', 'Bentinck',
             'Taats van Amerongen', 'Emden',
             'Raadtpenfionaris van Hoornbeeck', 'van Hoorn', 'Van Renswoude', 'Van Dam', 'Torck']

president_searcher = FuzzyKeywordSearcher(config=base_config)
president_searcher.index_keywords('PRASIDE')


