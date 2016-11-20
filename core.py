# http://www.nasdaq.com/quotes/
# https://www.google.com/finance/option_chain
# http://stackoverflow.com/questions/15198426/fixing-invalid-json-escape
# http://stackoverflow.com/questions/7917107/add-footnote-under-the-x-axis-using-matplotlib

import urllib.request
import re
import json
from collections import namedtuple
from datetime import date
from operator import mul
import matplotlib.pyplot as pyplot
from math import log, sqrt

SYMBOLS = "AAPL,TSLA,FB,MSFT,AKS,ETP,GOOG,AMZN,F,TWTR".split(",")


class Scraper():
    fixer = re.compile(r'(?<=[{,])([A-Za-z0-9_]*)(?=:)')
    invalid_escape = re.compile(r'\\x[0-7][0-9A-Fa-f]')

    def __init__(self):
        pass

    def build_url(self, sub, quary):
        if sub:
            sub = "/" + sub
        return "https://www.google.com/finance{}?{}&output=json".format(sub, build_quary(quary))

    def get_data(self, sub, **kwargs):
        url = self.build_url(sub, kwargs)
        print("fetching", url)
        with urllib.request.urlopen(url) as page:
            jsonstring = page.read().decode("UTF-8")

            jsonstring = Scraper.fixer.sub(lambda x: "\"{}\"".format(x.group(1)), jsonstring)
        if jsonstring.startswith("\n// "):
            jsonstring = jsonstring[4:]

        jsonstring = Scraper.invalid_escape.sub("?", jsonstring)
        return json.loads(jsonstring)

    def __call__(self, sub, **kwargs):
        return self.get_data(sub, **kwargs)


def build_quary(dict_):
    return "&".join("{}={}".format(k, v) for k, v in dict_.items())


def replace_with_byte(match):
    return chr(int(match.group(0)[2:], 8))


def print_json(json_):
    print(json.dumps(json_, indent=3))


# Option = namedtuple("Option","strike,maturity,price,bid,ask")

def strpnumber(text):
    if not text or text == "-":
        return 0
    return float(re.sub("[^0-9.]", "", text))


class OptionLoader():
    scraper = Scraper()

    def __init__(self, symbol):
        self.symbol = symbol
        self.name = OptionLoader.scraper("", q=self.symbol)[0]["name"]
        print("loading", self.name)

    def get_end_dates(self):
        for dict_ in (OptionLoader.scraper("option_chain", q=self.symbol).get("expirations", ())):
            yield extract_date(dict_)

    def get_options(self, type_, end_date):
        assert type_ in ("call", 'put'), "type should be 'call' or 'put'"
        info = OptionLoader.scraper("option_chain", q=self.symbol, **pack_date_custom(end_date)).get(type_ + "s", ())
        # for e in info:
        #    yield Option(e["strike"],(end_date-date.today()).days,e["p"],e["a"],e["b"])
        return {strpnumber(e["strike"]): strpnumber(e["p"]) for e in info if e["strike"] != "-" and e["p"] != "-"}

    def get_optionssets(self):
        for enddate in self.get_end_dates():
            yield OptionSet(self.name, self.symbol, enddate,
                            self.get_options("call", enddate), self.get_options("put", enddate))


class OptionSet():
    def __init__(self, name, nameshort, end_date: date, calls, puts):
        self.name = name
        self.descr = "{} -> {}". \
            format(date.today().strftime("%d %b %Y"), end_date.strftime("%d %b %Y"))
        self.short = nameshort + "_" + end_date.isoformat()
        self.calls = calls
        self.puts = puts
        self.maturity = (end_date - date.today()).days

    def putcall_parity(self):
        pass

    def export_plots(self):
        xc, yc = zip(*sorted(self.calls.items()))
        xp, yp = zip(*sorted(self.puts.items()))
        # print(xc,yc,xp,yp)
        pyplot.plot(xc, yc, "b", xp, yp, "r")
        pyplot.xlabel(r"strike price")
        pyplot.ylabel(r"option price")
        pyplot.legend(("call", "put"))
        pyplot.title(self.name)
        pyplot.annotate(self.descr, (0, 1), (5, -5), xycoords='axes fraction', textcoords='offset points', va='top',
                        ha='left')
        pyplot.savefig(self.short)
        pyplot.close()


def extract_date(dict_):
    return date(dict_["y"], dict_["m"], dict_["d"])


def pack_date(date_):
    return ({"y": date_.year, "m": date_.month, "d": date_.day})


def pack_date_custom(date_, prefix="exp"):
    return ({prefix + "y": date_.year, prefix + "m": date_.month, prefix + "d": date_.day})


def regression_coupled(x):
    return regression(*zip(*x))


def regression(x, y):
    N = len(x)
    assert len(y) == N, "Elements of x and y are not equal"
    Sx = sum(x)
    Sy = sum(y)
    Sxx = sum(map(mul, x, x))
    Sxy = sum(map(mul, x, y))
    Syy = sum(map(mul, y, y))
    b = (N * Sxy - Sx*Sy) / (N * Sxx - Sx ** 2)
    a = (Sy / N - b * Sx /N)
    s2 = 1 / N / (N - 2) * (N* Syy - Sy**2 - b**2 * (N*Sxx - Sx**2))
    sb = sqrt(N * s2 / (N*Sxx - Sx**2))
    sa = sqrt(sb - Sxx/N)
    return a, b, sa, sb


def plot_regression(title,filename, x, y):
    a, b, *_ = regression(x,y)
    ry = (a*x_ + b for x_ in x)
    pyplot.plot(x, y, "b+", x, ry, "r")
    pyplot.xlabel(r"strike price")
    pyplot.ylabel(r"price call-put")
    pyplot.title(title)
    pyplot.savefig(filename)


if __name__ == "__main__":
    o = OptionLoader("MSFT")
    for d in o.get_optionssets():
        d.export_plots()
