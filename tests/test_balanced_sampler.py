import pandas as pd
from realestate_spam.data.preprocessing import BalancedSampler

def test_balanced_sampler():
  df = pd.read_feather('tests/unit_test_display_names_df')

  N = 32

  sampler = BalancedSampler(df, 'CATEGORY', N, random_state=69)

  # Test total size
  total_samples = len(sampler.balanced_df)
  assert N - len(df['CATEGORY'].unique()) <= total_samples <= N, "Output DataFrame has incorrect size"

  # Test equal representation
  category_counts = sampler.balanced_df['CATEGORY'].value_counts()
  assert all(count == N // len(category_counts) for count in category_counts), "Categories are not represented equally"

  # Test representation of unique entries
  for category, group in df.groupby('CATEGORY'):
    unique_entries = group.drop_duplicates()
    if len(unique_entries) < N // len(category_counts):
      assert all(entry in sampler.balanced_df['DISPLAY_NAME'].values for entry in unique_entries['DISPLAY_NAME'].values), f"Not all unique entries in category {category} are represented"

  # Test no duplicates among samples for each category
  for category, group in sampler.balanced_df.groupby('CATEGORY'):
    if len(df[df['CATEGORY'] == category]) >= N // len(category_counts):
      assert group['DISPLAY_NAME'].nunique() == len(group), f"Duplicate entries found in category {category}"

  # Test generator
  assert len(list(sampler.generate_samples())) == len(sampler.balanced_df), "Generator yields incorrect number of samples"

  print('end of test_balanced_sampler')