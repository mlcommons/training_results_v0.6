import os

class Config():
  def __init__(self, tf_root):
    self.demo_dir = os.path.join(tf_root, 'demo')
    self.demo_tmp_dir = os.path.join(tf_root, '../demo_tmp')

    self.pb_dir = os.path.join(self.demo_dir, 'pb')
    if not os.path.exists(self.pb_dir):
      os.makedirs(self.pb_dir)
    self.fp32_optimized_graph = os.path.join(self.pb_dir, 'freezed_resnet50_opt.pb')
    self.int8_graph = os.path.join(self.pb_dir, 'int8_resnet50.pb')
    self.int8_graph_logged = os.path.join(self.pb_dir, 'int8_resnet50_logged.pb')
    self.int8_graph_freese = os.path.join(self.pb_dir, 'int8_resnet50_freese.pb')
    self.int8_graph_final = os.path.join(self.pb_dir, 'int8_resnet50_final.pb')

    self.accuracy_script = os.path.join(self.demo_dir, 'accuracy.py')
    self.benchmark_script = os.path.join(self.demo_dir, 'benchmark.py')
    self.quantize_script = os.path.join(self.demo_dir, 'quantize_graph.py')

    self.min_max_log = os.path.join(self.demo_dir, 'min_max.log')


  input_names = 'input'
  output_names = 'predict'

  def set_fp32_graph(self, pb):
    self.fp32_original_graph = pb

  def set_dataset(self, ds):
    self.imagenet_data = ds
