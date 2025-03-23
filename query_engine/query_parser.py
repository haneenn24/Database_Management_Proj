import re

def parse_sql_query(sql):
    """
    Parse a simplified SQL query into a dictionary format.
    Supports:
      - SELECT col1, col2
      - FROM table (ignored here)
      - WHERE conditions (with AND)
      - DISTINCT keyword
    """

    query_dict = {
        "select": [],
        "where": "",
        "distinct": False
    }

    # Normalize and remove excessive whitespace
    sql = sql.strip().replace('\n', ' ').replace('\t', ' ')
    sql = re.sub(r'\s+', ' ', sql).lower()

    # Check for DISTINCT
    if "select distinct" in sql:
        query_dict["distinct"] = True
        select_split = sql.split("select distinct")[1]
    else:
        select_split = sql.split("select")[1]

    # Split SELECT and WHERE
    if "where" in select_split:
        columns_part, where_part = select_split.split("where", 1)
        query_dict["where"] = where_part.strip()
    else:
        columns_part = select_split

    # Extract columns
    columns = columns_part.split("from")[0].strip()
    query_dict["select"] = [col.strip() for col in columns.split(",")]

    return query_dict


# Example queries to test
example1 = "SELECT Name, Department FROM Employees WHERE Salary > 6000 AND Department = 'HR'"
example2 = "SELECT DISTINCT Name FROM Employees WHERE Age < 30"

parsed1 = parse_sql_query(example1)
parsed2 = parse_sql_query(example2)

parsed1, parsed2
