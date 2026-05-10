"""Macro event labels shared by the presentation notebooks.

These events are used as known macro shock windows. They are not the target
for idiosyncratic changepoint detection; they help separate broad market moves
from stock-specific shocks.
"""

from __future__ import annotations

import sys

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd


KNOWN_EVENTS = (
    {
        "event": "May 2006 global risk selloff",
        "event_date": "2006-05-22",
        "event_type": "Global risk",
    },
    {
        "event": "Bear Stearns funds bankruptcy",
        "event_date": "2007-08-01",
        "event_type": "Credit",
    },
    {
        "event": "BNP Paribas freezes three funds",
        "event_date": "2007-08-09",
        "event_type": "Credit",
    },
    {
        "event": "Lehman bankruptcy",
        "event_date": "2008-09-15",
        "event_type": "Credit",
    },
    {
        "event": "GFC selloff peak",
        "event_date": "2008-10-10",
        "event_type": "Global risk",
    },
    {
        "event": "Flash crash",
        "event_date": "2010-05-06",
        "event_type": "Market structure",
    },
    {
        "event": "Euro sovereign crisis",
        "event_date": "2010-05-10",
        "event_type": "Sovereign",
    },
    {
        "event": "US downgrade / euro stress",
        "event_date": "2011-08-08",
        "event_type": "Sovereign",
    },
    {
        "event": "Draghi whatever it takes",
        "event_date": "2012-07-26",
        "event_type": "Policy",
    },
    {
        "event": "SNB removes EUR/CHF floor",
        "event_date": "2015-01-15",
        "event_type": "Policy",
    },
    {
        "event": "ECB QE launch",
        "event_date": "2015-01-22",
        "event_type": "Policy",
    },
    {
        "event": "Greek capital controls",
        "event_date": "2015-06-29",
        "event_type": "Sovereign",
    },
    {
        "event": "China Black Monday",
        "event_date": "2015-08-24",
        "event_type": "Global risk",
    },
    {
        "event": "Brexit vote",
        "event_date": "2016-06-24",
        "event_type": "Political",
    },
    {
        "event": "Trump election 2016",
        "event_date": "2016-11-09",
        "event_type": "Political",
    },
    {
        "event": "Volmageddon",
        "event_date": "2018-02-05",
        "event_type": "Market structure",
    },
    {
        "event": "Italy BTP selloff",
        "event_date": "2018-05-29",
        "event_type": "Sovereign",
    },
    {
        "event": "Q4 2018 selloff",
        "event_date": "2018-10-10",
        "event_type": "Global risk",
    },
    {
        "event": "COVID crash",
        "event_date": "2020-02-24",
        "event_type": "Pandemic",
    },
    {
        "event": "COVID recovery",
        "event_date": "2020-03-23",
        "event_type": "Pandemic",
    },
    {
        "event": "Rate-hike selloff",
        "event_date": "2022-01-05",
        "event_type": "Rates",
    },
    {
        "event": "Russia-Ukraine war",
        "event_date": "2022-02-24",
        "event_type": "Geopolitical",
    },
    {
        "event": "UK mini-budget",
        "event_date": "2022-09-26",
        "event_type": "Policy",
    },
    {
        "event": "SVB collapse",
        "event_date": "2023-03-10",
        "event_type": "Credit",
    },
    {
        "event": "Credit Suisse / UBS rescue",
        "event_date": "2023-03-15",
        "event_type": "Credit",
    },
    {
        "event": "Israel-Hamas war",
        "event_date": "2023-10-09",
        "event_type": "Geopolitical",
    },
    {
        "event": "Yen carry trade unwind",
        "event_date": "2024-08-05",
        "event_type": "Global risk",
    },
    {
        "event": "Liberation Day tariffs",
        "event_date": "2025-04-03",
        "event_type": "Policy",
    },
    {
        "event": "US-Iran war",
        "event_date": "2026-02-28",
        "event_type": "Geopolitical",
    },
)


def known_events_frame(start_date=None, end_date=None) -> pd.DataFrame:
    """Return known macro events as a tidy DataFrame."""
    events = pd.DataFrame(KNOWN_EVENTS)
    events["event_date"] = pd.to_datetime(events["event_date"])
    events = events.sort_values("event_date").reset_index(drop=True)

    if start_date is not None:
        events = events.loc[events["event_date"].ge(pd.Timestamp(start_date))]
    if end_date is not None:
        events = events.loc[events["event_date"].le(pd.Timestamp(end_date))]

    return events.reset_index(drop=True)
