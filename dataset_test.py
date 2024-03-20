
from tensorflow_datasets import testing
from tensorflow_datasets.datasets.speech_commands import speech_commands_dataset_builder


class SpeechCommandsTest(testing.DatasetBuilderTestCase):
  # TODO(speech_commands):
  DATASET_CLASS = speech_commands_dataset_builder.Builder
  SPLITS = {
      "train": 4,  # Number of fake train example
      "validation": 3,  # Number of fake validation example
      "test": 1,  # Number of fake test example
  }

  DL_EXTRACT_RESULT = ["train.tar.gz", "test.tar.gz"]


if __name__ == "__main__":
  testing.test_main()