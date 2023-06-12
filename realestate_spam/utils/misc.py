import html

def show_overlap(df1, df2, label=None, col=None, display_cols=[]):
  assert col is not None, 'col must be provided'
  if label is None:
    overlaps = set(df1[col].values).intersection(set(df2[col].values))
  else:
    overlaps = set(df1.q(f"class_label == '{label}'")[col].values).intersection(set(df2.q(f"class_label == '{label}'")[col].values))

  if None in overlaps:
    overlaps.remove(None)

  if label is None:
    return overlaps, df1.q(f"{col}.isin(@overlaps)")[display_cols], df2.q(f"{col}.isin(@overlaps)")[display_cols]
  else:
    return overlaps, df1.q(f"class_label == '{label}' and {col}.isin(@overlaps)")[display_cols], df2.q(f"class_label == '{label}' and {col}.isin(@overlaps)")[display_cols]

def html_unescape(df):
  df.NOTE = df.NOTE.apply(html.unescape)


