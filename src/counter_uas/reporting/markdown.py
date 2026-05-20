from __future__ import annotations

import pandas as pd


def dataframe_to_markdown(df: pd.DataFrame, include_index: bool = False) -> str:
    table = df.reset_index() if include_index else df.copy()
    columns = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format(value) for value in row) + " |")
    return "\n".join(lines)


def _format(value: object) -> str:
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.6g}"
    return str(value)
