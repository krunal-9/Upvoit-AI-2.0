import sqlparse

def format_sql(sql: str) -> str:
    if not sql:
        return ""
    return sqlparse.format(
        sql,
        reindent=True,
        keyword_case="upper",
        indent_width=2
    ).strip()
