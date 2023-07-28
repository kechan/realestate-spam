from realestate_spam.data.augment import TestAugmentedDataGenerator


def test_test(augs_df=None, reference_df=None):

  if augs_df is None: return

  aug_generator = TestAugmentedDataGenerator(augs_df, reference_df)
  aug_generator.apply_additional_filters("~NOTE_AUG.str.contains('water')")
  median_len = np.median(augs_df.NOTE_AUG.apply(len).values)
  aug_generator.apply_additional_filters(f"NOTE_AUG.str.len() <= {median_len}")
  aug_generator.generate_aug_df(10)