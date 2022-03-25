import os
from tensorflow.python.framework import tensor_util
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record

def my_summary_iterator(path):
	for r in tf_record.tf_record_iterator(path):
		yield event_pb2.Event.FromString(r)

base = "/home/kiadmin/projects/Interactive-Deep-Colorization-and-Compression/res/out/gen_compression_color/log_dir/fit"
latest = os.listdir(base)[-1]

path_to_latest = os.path.join(base, latest)

filename = os.listdir(path_to_latest)[0]

path_to_file = os.path.join(path_to_latest, filename)

for n, event in enumerate(my_summary_iterator(path_to_file)):
	if n % 999 == 0:
		for value in event.summary.value:
			t = tensor_util.MakeNdarray(value.tensor)
			print(value.tag, event.step, t)
