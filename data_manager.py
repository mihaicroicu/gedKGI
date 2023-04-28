import pandas as pd
from pathlib import Path

from gedAPI import GedLoader
from ingester3.ViewsMonth import ViewsMonth
from typing import Union


def newest_full_ged(views_month: ViewsMonth):
    """
    Given a year and month pair, decide what is the newest full ged version
    :param views_month: A ViewsMonth object containing the month you're interested in.
    :return: The nearest real GED version
    """
    for i in list(reversed(range(17, views_month.year - 2000 + 1))):
        try_ged = f"{i}.1"
        if GedLoader(version=try_ged, verbose=False).exists:
            return try_ged

    raise ValueError(f"""You are trying to rebase against a much too old GED
                     or something really bad is going on with the GED API! 
                     Please report!""")


def month_id_to_ged_list(month_id: Union[int, None] = None,
                         rebase_against: Union[str, 'newest', 'time_machine'] = 'newest') -> list[GedLoader]:
    """
    Given a month id construct a list of all the GED API calls needed to reach
    this month_id from a given "True" GED all the way to the given month.
    E.g. if GED max is 22.1 (1989-2021.12) and current month is 2023-02
    import that and all the candidates from GED 22.0.1 to 23.0.2
    :param month_id: ViEWS month_id. If None, use now
    :param rebase_against: Explicit version of GED to rebase against.
    If time_machine, it will revert to building the API call as if we were in that month, ignoring all full GEDs
    released later than that month. If newest, it will use the newest GED full and ignore any candidates that
    are also covered by time_machine. Default is newest.
    :return: A list of gedAPIs that need to be imported and filtered
    """
    if month_id is None:
        month_id = ViewsMonth.now()
    else:
        month_id = ViewsMonth(month_id)

    base_ged = None

    if rebase_against == 'newest':
        base_ged = GedLoader(newest_full_ged(ViewsMonth.now()))

    if rebase_against == 'time_machine':
        base_ged = GedLoader(newest_full_ged(month_id))

    if base_ged is None:
        base_ged = GedLoader(rebase_against)
        if not base_ged.exists:
            raise KeyError(f"Ged {rebase_against} does not exist!")

    if base_ged.max_month >= month_id:
        return [base_ged]

    parsed_list = [base_ged]
    for i in range(base_ged.max_month.id + 1, month_id.id + 1):
        cur_cand = GedLoader(f"{ViewsMonth(i).year - 2000}.0.{ViewsMonth(i).month}")
        if cur_cand.exists:
            parsed_list += [cur_cand]

    return parsed_list


def fetch_from_list(parsed_list):
    loaded_ged = parsed_list.copy()
    [i.load() for i in loaded_ged]
    ged = pd.concat([i.ged for i in loaded_ged], ignore_index=True)
    return ged


def ged_consolidated_from_disk_or_api(month_id=None, how='newest', path='~/.ged_loader_cache/', verbose=False):

    Path(path).mkdir(parents=True, exist_ok=True)

    if month_id is None:
        month_id = ViewsMonth.now().id

    file_path = Path(path,f'ged_{how}_{month_id}.parquet')
    if verbose:
        print ("Saving to", file_path)

    try:
        ged = pd.read_parquet(file_path)
        print ("File found, no re-fetch needed!")
    except FileNotFoundError:
        if verbose:
            print ("File not found... Fetching data from API")
        ged = fetch_from_list(month_id_to_ged_list(month_id=month_id, rebase_against=how))
        ged.to_parquet(file_path)
    return ged


if __name__ == "__main__":
    print(f"Fetching data from {ViewsMonth.from_year_month(1989,1).start_date} to {ViewsMonth.now().end_date}.")
    df = ged_consolidated_from_disk_or_api(verbose=True)
    print (f"Consolidated GED is a dataframe of shape {df.shape} and is saved locally.")