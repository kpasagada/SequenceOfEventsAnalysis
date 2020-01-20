Command to run the solution:
../../Documents/spark/bin/spark-submit --master local[2] event_prediction.py <hdfs-input-file-path>/event_combined.json > <output-file>

Sample command:
../../Documents/spark/bin/spark-submit --master local[2] event_prediction.py hdfs://localhost:9000/project/input/event_combined.json > output.txt

Files present:

- data_extraction.py: used to extract event data for a given month and year.
- data_combiner.py: used to combine these data into a single JSONL format file.
- event_prediction.py: pyspark solution for root_code prediction.
- event_combined.json: train and test data
- output.txt: output of event_prediction.py

This solution was run within PyCharm terminal.