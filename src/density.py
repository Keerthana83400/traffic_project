def classify(count):
    if count <= 5:  return "LOW"       # show this table in your report
    if count <= 15: return "MEDIUM"
    return "HIGH"