---
name: market-data
description: Use only for current stock/share prices, ticker quotes, and financial market movers (gainers, losers). Network switch must be on; no API key.
---

# Market data

One script. It asks a JSON endpoint for rows that are already typed and ranked,
so nothing has to be parsed out of a page or remembered.

```
python3 scripts/market_movers.py --movers gainers --count 10
python3 scripts/market_movers.py --movers losers --count 5
python3 scripts/market_movers.py --movers actives --count 10
python3 scripts/market_movers.py --quote AAPL MSFT NVDA
```

It prints a heading naming the source and the data's own timestamp, then a
markdown table. **That output is the answer**: copy the rows as printed. Do not
round them, re-order them, or add a column the table does not have.

| Option | Effect |
| --- | --- |
| `--movers gainers\|losers\|actives` | the day's movers, ranked by the endpoint |
| `--quote SYM ...` | quote named symbols instead |
| `--count N` | how many rows, 1-50 (default 10) |
| `--any-instrument` | include funds and trusts, not only ordinary shares |
| `--json FILE` | also write the rows as objects, for a follow-up script |
| `--spec FILE` | also write a `make_pptx.py` spec for these rows |
| `--title TEXT` | title for `--spec` |

## Putting it in a deck

`--spec` writes a spec the documents skill accepts as it stands, so the whole
request is two commands:

```
python3 scripts/market_movers.py --movers gainers --count 10 --spec spec.json --title "Top 10 gainers today"
python3 ../documents/scripts/make_pptx.py --spec spec.json --out gainers.pptx
```

The spec splits at twelve rows per slide, which is what a table slide holds.
Return the `.pptx` to the user as the downloadable artifact.

## What it will not do

**It never fills a gap.** A row has to carry a symbol, a name, a price, a
change, a percentage and a volume, with numbers where numbers belong. A row
missing any of them is dropped, so asking for ten can return eight — and eight
real rows is the answer, not a reason to invent two. If nothing usable comes
back the script exits non-zero and prints why; say that, and do not answer the
question from memory. Prices you remember are wrong by definition.

**It does not say why a price moved.** The response carries prices, not
reasons. A number here supports "ROIV is up 18.75% today"; it supports no
sentence containing "because", "on news of", or "driven by". If the user asks
why, say the data does not carry it.

**Read the Currency column.** `--movers` is US shares and prints USD, but
`--quote` takes any symbol the source knows, and those are priced in their own
market's unit — a London line comes back in `GBp`, which is PENCE, so 1,574.80
is £15.75 and not £1,574.80. The unit is a column in the table for that reason.
Never compare two rows' prices without it, and never drop it when you quote a row.

**Read the heading too.** It says when the rows were stamped, and it says
"rows stamped between" when they do not share a moment, which they will not
across exchanges. If the session is not open it says so — "last market closed
price" — and then "today" in a title means the last close, not a live price.

**A long `--quote` can run out of time.** Each symbol is its own request, and the
run keeps a budget inside the tool's timeout so a few slow symbols cannot cost you
the whole answer. If it runs out, the table holds the symbols that answered and a
note names the ones left out; ask for those in a second run rather than repeating
the whole list.

**One moment, not a history.** These are prices as the source last stamped them.
It is not a portfolio, a history, a forecast, or advice, and nothing here should
be presented as any of those.

**Where the data comes from.** Two JSON endpoints Yahoo publishes for its own
front end. They are undocumented and unofficial: they can change or start
refusing without notice, which is what the non-zero exits are for. Attribute the
source in your answer, as the printed heading does. Do not present the figures as
a licensed market feed, and do not build anything that depends on them staying
available.
