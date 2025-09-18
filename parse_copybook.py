import re

def parse_copybook(copybook_path):
    """
    Parse COBOL copybook and return list of (field_name, length, type).
    Supports PIC X (alphanumeric) and PIC 9 (numeric).
    """
    fields = []
    with open(copybook_path, "r") as f:
        for line in f:
            line = line.strip()
            match = re.search(r'(\w[\w-]*)\s+PIC\s+([X9])\((\d+)\)', line, re.IGNORECASE)
            if match:
                field_name = match.group(1).replace("-", "_")
                field_type = "STRING" if match.group(2).upper() == "X" else "NUMERIC"
                field_len = int(match.group(3))
                fields.append((field_name, field_len, field_type))
    return fields

# Example usage:
#schema = parse_copybook("employee.cpy")
#print(schema)
# [('EMP_ID', 5, 'STRING'), ('EMP_NAME', 15, 'STRING'), ('EMP_DEPT', 5, 'STRING'), ('EMP_SALARY', 10, 'NUMERIC')]


def build_schema_with_positions(fields):
    """
    Takes [(field, length, type)] and returns [(field, start, end, type)]
    """
    schema = []
    pos = 1
    for name, length, ftype in fields:
        schema.append((name, pos, pos + length - 1, ftype))
        pos += length
    return schema

# Example
#fields = [('EMP_ID', 5, 'STRING'), ('EMP_NAME', 15, 'STRING')]
#schema_with_pos = build_schema_with_positions(fields)
#print(schema_with_pos)
# [('EMP_ID', 1, 5, 'STRING'), ('EMP_NAME', 6, 20, 'STRING')]
