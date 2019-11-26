province_order = ["gelderlandt",
                  "hollandt ende west-frieslandt",
                  "utrecht",
                  "frieslandt",
                  "overijssel",
                  "groningen",
                  "zeelandt"
                 ]


searchstring = "met {} extraordinaris gedeputeerde uyt de provincie van {}"
provinces = []
for provincie in province_order:
    for telwoord in ['een', 'twee', 'drie']:
        provinces.append(searchstring.format(telwoord, provincie))

naaminput = ['Van Maasdam', 'Cammingha', 'Haersolte', 'Iddekinge', 'Isselmuden', 'van Welderen',
             'Ockersse', 'Heukelum', 'Rose', 'Rouse', 'van Singendonck', 'Zanders', 'Bentinck',
             'Taats van Amerongen', 'Emden',
             'Raadtpenfionaris van Hoornbeeck', 'van Hoorn', 'Van Renswoude', 'Van Dam', 'Torck']

president_searcher = FuzzyKeywordSearcher(config=base_config)
president_searcher.index_keywords('PRASIDE')
alternatives = []
for T in presentielijsten:
    president = president_searcher.find_candidates_new(keyword='PRASIDE', text=T.text)
    if president:
        alternatives.append(president[0]['match_string'])


