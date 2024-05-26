#!/usr/bin/env python3

import random

import click
import numpy as np
import pandas as pd


def add_month_to_ym(target_dateym: int, add_months: int) -> int:
    yyyy, mm = divmod(target_dateym, 100)
    mm += add_months
    while True:
        if 1 <= mm <= 12:
            break

        if mm < 1:
            mm += 12
            yyyy -= 1

        if mm > 12:
            mm -= 12
            yyyy += 1

    return yyyy * 100 + mm


def calculate_diff_month_with_ym(from_dateym: int, to_dateym: int) -> int:
    yyyy_from, mm_from = divmod(from_dateym, 100)
    yyyy_to, mm_to = divmod(to_dateym, 100)
    return (yyyy_to - yyyy_from) * 12 + (mm_to - mm_from)


def is_rebalance_timing(
    start_dateym,
    current_dateym,
    month_of_rebalance_frequency: int,
) -> bool:
    diff_month = calculate_diff_month_with_ym(start_dateym, current_dateym)
    return diff_month % month_of_rebalance_frequency == 0


def read_universe(current_dateym: int) -> pd.DataFrame:
    size = 4000
    np.random.seed(current_dateym)
    nri_code_l = [f"{i:0>5}" for i in range(size)]
    df = pd.DataFrame(
        {
            "NRI_CODE": nri_code_l,
            "STOCK_FLG": "",
        },
    )
    drop_nri_code_l = random.sample(nri_code_l, int(np.random.uniform(size) / 10))
    mask = df["NRI_CODE"].isin(drop_nri_code_l)
    return df[~mask]


def read_future_return(current_dateym: int) -> pd.DataFrame:
    size = 4000
    np.random.seed(current_dateym)
    nri_code_l = [f"{i:0>5}" for i in range(size)]
    df = pd.DataFrame(
        {
            "NRI_CODE": nri_code_l,
            "STOCK_FLG": "",
            "DRTNF1": np.random.randn(size) / 25,
        },
    )
    drop_nri_code_l = random.sample(nri_code_l, int(np.random.uniform(size) / 10))
    mask = df["NRI_CODE"].isin(drop_nri_code_l)
    return df[~mask]


def read_factor(current_dateym: int, factor_name: str) -> pd.DataFrame:
    size = 4000
    np.random.seed(current_dateym)
    nri_code_l = [f"{i:0>5}" for i in range(size)]
    df = pd.DataFrame(
        {
            "NRI_CODE": nri_code_l,
            "STOCK_FLG": "",
            f"{factor_name}": np.random.randn(size) / 25,
        },
    )
    drop_nri_code_l = random.sample(nri_code_l, int(np.random.uniform(size) / 10))
    mask = df["NRI_CODE"].isin(drop_nri_code_l)
    return df[~mask]


@click.command()
@click.argument("from_dateym", type=int)
@click.argument("to_dateym", type=int)
@click.argument("factor_name", type=str)
@click.argument("month_of_rebalance_frequency", type=int)
def main(
    from_dateym: int,
    to_dateym: int,
    factor_name: str,
    month_of_rebalance_frequency: int,
):
    current_dateym = from_dateym
    while current_dateym <= to_dateym:
        # 将来1カ月のリターン情報を読み込み
        future_return_df = read_future_return(current_dateym)
        if is_rebalance_timing(
            from_dateym,
            current_dateym,
            month_of_rebalance_frequency,
        ):
            print("reb")
            # リバランスタイミングならポートフォリオ作成
            # universeの読み込み
            universe_df = read_universe(current_dateym)
            # factorの読み込み
            ## size
            size_df = read_factor(current_dateym, "size")
            ## factor
            factor_df = read_factor(current_dateym, factor_name)

            universe_df = universe_df.merge(
                future_return_df,
                how="inner",
                on=["NRI_CODE", "STOCK_FLG"],
            )
            universe_df = universe_df.merge(
                size_df,
                how="inner",
                on=["NRI_CODE", "STOCK_FLG"],
            )
            universe_df = universe_df.merge(
                factor_df,
                how="inner",
                on=["NRI_CODE", "STOCK_FLG"],
            )

            # sizeで2分位, factorで3分位
            universe_df = universe_df.dropna(how="any", axis="index")
            universe_df = universe_df.assign(
                SIZE_G=pd.qcut(universe_df["size"], 2, labels=["small", "big"]),
            )
            universe_df = universe_df.assign(
                FACTOR_G=universe_df.groupby(["SIZE_G"], observed=False)[
                    factor_name
                ].transform(
                    lambda x: pd.qcut(x, 3, labels=["S", "M", "L"]),
                ),
            )
            # weightの計算
            universe_df = universe_df.assign(
                market_value=universe_df["size"].abs(),
            )
            universe_df = universe_df.assign(
                weight=universe_df.groupby(["SIZE_G", "FACTOR_G"], observed=False)[
                    "market_value"
                ].transform(
                    lambda s: s.div(s.sum()),
                ),
            )
            # print(universe_df)
            # print(universe_df["weight"].sum())

        else:
            print("not reb")
            # 将来リターンがない場合はリバランス
            universe_df = universe_df.merge(
                future_return_df,
                how="left",
                on=["NRI_CODE", "STOCK_FLG"],
            )
            universe_df = universe_df.dropna(how="any", axis="index")
            universe_df = universe_df.assign(
                weight=universe_df.groupby(["SIZE_G", "FACTOR_G"], observed=False)[
                    "weight"
                ].transform(
                    lambda s: s.div(s.sum()),
                ),
            )

        # 1カ月間運用
        universe_df = universe_df.assign(
            DW=universe_df["weight"].mul(universe_df["DRTNF1"]),
        )
        print(universe_df.groupby(["SIZE_G", "FACTOR_G"], observed=False)["DW"].sum())
        ## 新ウェイト
        universe_df = universe_df.assign(
            weight=universe_df["weight"].mul(universe_df["DRTNF1"].add(1)),
        )
        del universe_df["DRTNF1"]
        universe_df = universe_df.assign(
            weight=universe_df.groupby(["SIZE_G", "FACTOR_G"], observed=False)[
                "weight"
            ].transform(
                lambda s: s.div(s.sum()),
            ),
        )
        # logger.info("%d end!", current_dateym)
        print(current_dateym)
        current_dateym = add_month_to_ym(current_dateym, 1)


if __name__ == "__main__":
    main()
