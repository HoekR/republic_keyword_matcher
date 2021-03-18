from string import Template
import pandas as pd

query= Template('''SELECT raa_new.functie.naam as functienaam, 
	   raa_new.instelling.naam as college,
       raa_new.persoon.searchable as regent,
       raa_new.aanstelling.*
FROM raa_new.aanstelling 
JOIN raa_new.functie on raa_new.aanstelling.functie_id = raa_new.functie.id
JOIN raa_new.instelling on raa_new.aanstelling.instelling_id = raa_new.instelling.id
JOIN raa_new.persoon on raa_new.aanstelling.persoon_id = raa_new.persoon.id
WHERE raa_new.instelling.naam LIKE "%Staten-Generaal%"
AND raa_new.aanstelling.persoon_id in ($ids);''')

class DisambiguateDelegate(object):
    def __init__(self,
                 connection = "mysql+pymysql://rik:X0chi@localhost/raa_new",
                 query=query,
                 df=None):
        self.con = connection
        self.query = query

    def repair_date(self, row, item):
        if pd.isna(row[item]):
            item = item + "_als_bekend"
        result = pd.Period(row[item], freq="D")
        return result

    def daystoperiod(self, row):
        try:
            if pd.isna(row.van_p):
                if pd.isna(row.tot_p):
                    raise ValueError("no period without dates")
                else:
                    start = row.tot_p - 365 # this may be a day off for leap years, but ...
            else:
                start = row.van_p
            if pd.isna(row.tot_p):
                if pd.isna(start):
                    raise ValueError("no period without dates")
                else:
                    end = row.van_p + 365
            else:
                end = row.tot_p
            return pd.period_range(start=start, end=end, freq="D")
        except ValueError:
            return pd.NA

    def best_period_match(self, date, df=None):
        exact = True
        df = df.loc[df.p_interv.notna()]
        result = df.loc[df.p_interv.apply(lambda x: date in x)]
        if len(result) == 0:
            result = df.sort_values('daycompare')
            exact = False
        return {"id":result.id.iat[0], "exact":exact, "row": result.head(1)}

    def closerdate(self, date, period):
        if date < period.min():
            result = period.min() - date
        else:
            result = date - period.max()
        return result

    def closer_best_match(self, name="", date="",query=query):
        p_date = pd.Period(date, "D")
        hs = identify(name=name, year=p_date.year,fuzzy=False, window=50, df=abbreviated_delegates)
        persoonids = hs.id.unique()
        list(persoonids)
        sids = [f"{persoonid}" for persoonid in persoonids]
        sids = ",".join(sids)
        if len(sids) == 0:
            return None, "no values"
        q = query.substitute(ids=sids)
        #print (q)
        distinguisdf = pd.read_sql(con=self.con, sql=text(q), parse_dates={'van':None,
                                                            'tot':None})
        for item in ['van', 'tot']:
            distinguisdf[item + '_p'] = distinguisdf.apply(lambda row: self.repair_date(row, item), axis=1)
    #     distinguisdf['van_p'] = distinguisdf.van.apply(lambda x: pd.Period(x, freq="D")) # this should be possible with to_period, but gives an error
    #     distinguisdf['tot_p'] = distinguisdf.tot.apply(lambda x: pd.Period(x, freq="D"))
        distinguisdf["p_interv"] = distinguisdf.apply(lambda row: self.daystoperiod(row), axis=1)
        distinguisdf['daycompare'] = distinguisdf.p_interv.apply(lambda x: self.closerdate(p_date, x))
        result = self.best_period_match(date=p_date, df=distinguisdf)
        if len(result)>0:
            return result
        else:
            return None

cdksearcher = FuzzyKeywordSearcher(config=fuzzysearch_config)

def dedup_candidates(proposed_candidates=[],
                     searcher=FuzzyKeywordSearcher,
                     register=dict,
                     dayinterval=pd.Interval,
                     df=pd.DataFrame,
                     searchterm=''):
    scores = {}
    for d in proposed_candidates:
        prts = nm_to_delen(d)
        for p in prts:
            if p != searchterm:
                score = cdksearcher.find_candidates(text=p)
                if len(score) > 1:
                    score=max(score, key=lambda x: x.get('levenshtein_distance'))
                if score:
                    try:
                        scores[d] = (score[0].get('levenshtein_distance'), p)
                    except:
                        print (score)
    if not scores:
        candidate = proposed_candidates
    else:
        candidate = [max(scores)]
        register[p] = scores[candidate[0]][1]
    result = dedup_candidates2(proposed_candidates=candidate, dayinterval=dayinterval, df=df)
    if len( result) == 0:
        candidate = pd.DataFrame() # searchterm
    return result

def dedup_candidates2(proposed_candidates=[], dayinterval=pd.Interval, df=pd.DataFrame):
    nlst = proposed_candidates
    res = df.loc[df.name.isin(proposed_candidates)]
    if len(res)>1:
        res = res[res.h_life.apply(lambda x: x.overlaps(dayinterval))]
    if len(res)>1:
        res = res[res.p_interval.apply(lambda x: x.overlaps(dayinterval))]
    if len(res)>1:
        res = res.loc[res.sg == True]
    return res