#this should go into a separate module
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
                 query=query):
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

    def best_period_match(self, date, df=distinguisdf):
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
        distinguisdf = pd.read_sql(con=self.con, sql=text(q), parse_dates={'van':None,
                                                            'tot':None})
        for item in ['van', 'tot']:
            distinguisdf[item + '_p'] = distinguisdf.apply(lambda row: repair_date(row, item), axis=1)
        distinguisdf["p_interv"] = distinguisdf.apply(lambda row: self.daystoperiod(row), axis=1)
        distinguisdf['daycompare'] = p_interv.apply(lambda x: self.closerdate(p_date, x))
        result = self.best_period_match(date=p_date, df=distinguisdf)
        if len(result)>0:
            return result
        else:
            return None