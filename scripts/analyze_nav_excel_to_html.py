from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fincore.report import create_strategy_report

DATE_CANDIDATES = [
    "日期",
    "交易日期",
    "净值日期",
    "date",
]

NAV_CANDIDATES = [
    "累计净值",
    "收盘价整体净值",
    "结算价整体净值",
    "整体净值",
    "累计单位净值",
    "单位净值",
    "复权净值",
    "收盘价净值",
    "结算价净值",
    "净值",
]


def normalize_name(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", "", text).lower()


def build_column_lookup(columns: pd.Index) -> dict[str, str]:
    return {normalize_name(column): str(column) for column in columns}


def resolve_column(
    lookup: dict[str, str],
    explicit_name: str | None,
    candidates: list[str],
    label: str,
) -> str:
    if explicit_name:
        explicit_key = normalize_name(explicit_name)
        if explicit_key in lookup:
            return lookup[explicit_key]
        raise ValueError(f"未找到指定的{label}列: {explicit_name}")

    for candidate in candidates:
        candidate_key = normalize_name(candidate)
        if candidate_key in lookup:
            return lookup[candidate_key]

    for candidate in candidates:
        candidate_key = normalize_name(candidate)
        for normalized_name, original_name in lookup.items():
            if candidate_key in normalized_name:
                return original_name

    raise ValueError(f"无法自动识别{label}列，可通过参数显式指定")


def score_sheet(sheet_name: str, nav_column: str, row_count: int) -> int:
    sheet_key = normalize_name(sheet_name)
    nav_key = normalize_name(nav_column)
    score = row_count
    # 优先选择累计净值列
    if nav_key == normalize_name("累计净值"):
        score += 6000
    elif nav_key == normalize_name("收盘价整体净值"):
        score += 5000
    elif nav_key == normalize_name("结算价整体净值"):
        score += 4000
    elif "整体净值" in nav_key:
        score += 3000
    elif "累计净值" in nav_key:
        score += 2000
    elif "净值" in nav_key:
        score += 1000
    if "收盘价" in sheet_key:
        score += 300
    elif "结算价" in sheet_key:
        score += 200
    return score


def load_sheet_candidate(
    excel_path: Path,
    sheet_name: str,
    date_column: str | None,
    nav_column: str | None,
) -> dict[str, object]:
    frame = pd.read_excel(excel_path, sheet_name=sheet_name)
    frame = frame.dropna(axis=1, how="all")
    lookup = build_column_lookup(frame.columns)
    original_date_column = resolve_column(lookup, date_column, DATE_CANDIDATES, "日期")
    original_nav_column = resolve_column(lookup, nav_column, NAV_CANDIDATES, "净值")

    cleaned = frame[[original_date_column, original_nav_column]].copy()
    cleaned.columns = ["date", "nav"]
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["nav"] = pd.to_numeric(cleaned["nav"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date", "nav"])
    cleaned = cleaned[cleaned["nav"] > 0]
    cleaned = cleaned.drop_duplicates(subset=["date"], keep="last")
    cleaned = cleaned.sort_values("date")
    if len(cleaned) < 2:
        raise ValueError(f"工作表 {sheet_name} 清洗后有效净值数据不足 2 行")

    return {
        "sheet_name": sheet_name,
        "date_column": original_date_column,
        "nav_column": original_nav_column,
        "data": cleaned,
        "score": score_sheet(sheet_name, original_nav_column, len(cleaned)),
    }


def load_nav_series(
    excel_path: Path,
    sheet_name: str | None,
    date_column: str | None,
    nav_column: str | None,
) -> tuple[pd.Series, dict[str, object]]:
    excel_file = pd.ExcelFile(excel_path)
    candidates: list[dict[str, object]] = []
    errors: list[str] = []
    target_sheets = [sheet_name] if sheet_name else excel_file.sheet_names

    for current_sheet in target_sheets:
        try:
            candidate = load_sheet_candidate(excel_path, current_sheet, date_column, nav_column)
            candidates.append(candidate)
        except Exception as exc:
            errors.append(f"{current_sheet}: {exc}")

    if not candidates:
        detail = "\n".join(errors) if errors else "未读取到任何可用工作表"
        raise ValueError(f"无法从 Excel 中提取净值序列:\n{detail}")

    best = max(candidates, key=lambda item: int(item["score"]))
    nav_series = best["data"].set_index("date")["nav"]
    nav_series.index.name = None
    nav_series.name = str(best["nav_column"])
    return nav_series, best


def resample_nav(nav_series: pd.Series, freq: str) -> pd.Series:
    """Resample NAV series to specified frequency (daily/weekly/monthly).

    Parameters
    ----------
    nav_series : pd.Series
        Daily NAV series indexed by date.
    freq : str
        'daily', 'weekly', or 'monthly'.

    Returns
    -------
    pd.Series
        Resampled NAV series.
    """
    if freq == "daily":
        return nav_series
    if freq == "weekly":
        # 取每周最后一个交易日净值
        return nav_series.resample("W-FRI").last().dropna()
    if freq == "monthly":
        # 取每月最后一个交易日净值
        return nav_series.resample("ME").last().dropna()
    raise ValueError(f"不支持的频率: {freq}，可选: daily/weekly/monthly")


def build_returns(nav_series: pd.Series) -> pd.Series:
    returns = nav_series.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2:
        raise ValueError("净值序列生成收益率后数据不足 2 行")
    returns.name = "strategy"
    return returns


def filter_nav_series(
    nav_series: pd.Series,
    start_date: str | None,
    end_date: str | None,
) -> pd.Series:
    filtered = nav_series
    if start_date:
        start_ts = pd.to_datetime(start_date)
        filtered = filtered[filtered.index >= start_ts]
    if end_date:
        end_ts = pd.to_datetime(end_date)
        filtered = filtered[filtered.index <= end_ts]
    if len(filtered) < 3:
        raise ValueError("按日期筛选后净值数据不足 3 行，无法生成报告")
    return filtered


def parse_args() -> argparse.Namespace:
    default_excel = Path(__file__).resolve().parents[1] / "累计净值数据_周度.xlsx"
    parser = argparse.ArgumentParser(description="从基金净值 Excel 生成 fincore HTML 业绩报告")
    parser.add_argument("excel_path", nargs="?", default=str(default_excel), help="Excel 文件路径")
    parser.add_argument("--output", help="输出 HTML 路径")
    parser.add_argument("--sheet-name", help="指定工作表名，不指定则自动选择")
    parser.add_argument("--date-column", help="指定日期列名")
    parser.add_argument("--nav-column", help="指定净值列名")
    parser.add_argument("--start-date", help="分析起始日期，例如 2021-01-01")
    parser.add_argument("--end-date", help="分析结束日期，例如 2022-12-31")
    parser.add_argument("--title", help="报告标题")
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=13,
        help="滚动窗口，默认13周",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    excel_path = Path(args.excel_path).expanduser().resolve()
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {excel_path}")

    output_path = Path(args.output).expanduser().resolve() if args.output else excel_path.with_suffix(".report.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nav_series, selected = load_nav_series(
        excel_path=excel_path,
        sheet_name=args.sheet_name,
        date_column=args.date_column,
        nav_column=args.nav_column,
    )
    nav_series = filter_nav_series(nav_series, args.start_date, args.end_date)

    # 数据已经是周度，直接计算收益率
    returns = build_returns(nav_series)
    rolling_window = args.rolling_window

    title = args.title or f"{excel_path.stem} 业绩分析报告"
    report_path = create_strategy_report(
        returns,
        title=title,
        output=str(output_path),
        rolling_window=rolling_window,
        period="weekly",
    )

    print(f"Excel: {excel_path}")
    print(f"Sheet: {selected['sheet_name']}")
    print(f"Date column: {selected['date_column']}")
    print(f"NAV column: {selected['nav_column']}")
    print(f"NAV rows: {len(nav_series)}")
    print(f"Date range: {nav_series.index.min():%Y-%m-%d} -> {nav_series.index.max():%Y-%m-%d}")
    print(f"Returns rows: {len(returns)}")
    print(f"Rolling window: {rolling_window}")
    print(f"HTML report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
